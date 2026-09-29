//! One request on one site, or on several at once with a shared
//! ranking (M5): discover → allowlist → route → validate
//! → re-discover and re-check the capability → execute → adapt. Every step is
//! recorded in the [`Trace`].

use std::time::Duration;

use futures::future::join_all;

use serde::Serialize;
use serde_json::{Map, Value};
use tokio::time::{Instant, timeout_at};

use crate::adapters::{self, AdapterError, Output, Source};
use crate::browser::{BrowserError, ChromeBrowser, Tab, TabOptions};
use crate::policy::{PolicyViolation, Site};
use crate::rank::{self, Constraints, Ranking};
use crate::router::{Router, RouterError};
use crate::schema::{ToolSpec, schema_hash};
use crate::trace::{self, Event, RankedRef, ToolRef, Trace};

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    Browser(#[from] BrowserError),
    #[error(transparent)]
    Router(#[from] RouterError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error("site timed out after {0:?}")]
    Timeout(Duration),
}

#[derive(Debug, Clone)]
pub struct AskOptions {
    /// How long the page may take to register its tools.
    pub wait: Duration,
    pub call_timeout: Duration,
    /// Budget for a whole site, from opening the tab to the adapted output.
    pub site_timeout: Duration,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            wait: Duration::from_secs(15),
            call_timeout: Duration::from_secs(30),
            site_timeout: Duration::from_secs(90),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AskOutcome {
    /// No allowlisted tool fits the request; nothing executed.
    NoTool,
    /// The chosen arguments fail the site schema; nothing executed.
    Invalid { tool: String, errors: Vec<String> },
    /// The policy refused the call; nothing executed.
    Blocked {
        tool: Option<String>,
        reason: String,
    },
    Done {
        tool: String,
        args: Map<String, Value>,
        output: Output,
    },
}

impl AskOutcome {
    pub fn status(&self) -> &'static str {
        match self {
            Self::NoTool => "no_tool",
            Self::Invalid { .. } => "invalid",
            Self::Blocked { .. } => "blocked",
            Self::Done { .. } => "done",
        }
    }
}

/// Answers `request` on `site` in a new tab of `browser`.
pub async fn ask_site(
    browser: &ChromeBrowser,
    site: &Site,
    request: &str,
    router: &Router,
    opts: &AskOptions,
    trace: &mut Trace,
) -> Result<AskOutcome, RunError> {
    trace.tag(&site.name);
    trace.record(Event::RunStarted {
        url: site.url.clone(),
        request: request.to_owned(),
    });
    let tab_options = TabOptions {
        legacy_shim: site.legacy_shim,
    };
    let deadline = Instant::now() + opts.site_timeout;
    let timed_out = || RunError::Timeout(opts.site_timeout);
    let result = match timeout_at(deadline, browser.open(&site.url, &tab_options)).await {
        Ok(Ok(tab)) => {
            let result = timeout_at(deadline, ask_tab(&tab, site, request, router, opts, trace))
                .await
                .unwrap_or_else(|_| Err(timed_out()));
            // Best effort: the browser is shut down by the caller anyway.
            let _ = tab.close().await;
            result
        }
        Ok(Err(e)) => Err(e.into()),
        Err(_) => Err(timed_out()),
    };
    trace.record(Event::RunFinished {
        status: result
            .as_ref()
            .map_or("error", AskOutcome::status)
            .to_owned(),
        error: result.as_ref().err().map(ToString::to_string),
    });
    result
}

/// One site's part of a multi-site run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SiteRun {
    pub site: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<AskOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiOutcome {
    pub sites: Vec<SiteRun>,
    pub ranking: Ranking,
}

/// Answers `request` on every site at once, each in its own tab and within
/// `opts.site_timeout`. A failing site leaves the other results intact; the
/// products of all sites are ranked together.
pub async fn ask_sites(
    browser: &ChromeBrowser,
    sites: &[&Site],
    request: &str,
    router: &Router,
    opts: &AskOptions,
    trace: &mut Trace,
) -> MultiOutcome {
    let mut children: Vec<Trace> = sites.iter().map(|s| trace.child(&s.name)).collect();
    let results = join_all(
        sites
            .iter()
            .zip(children.iter_mut())
            .map(|(site, child)| ask_site(browser, site, request, router, opts, child)),
    )
    .await;
    trace.absorb(children);

    let mut products = Vec::new();
    // The arguments the router chose, often an English rewrite of the
    // request, count as keywords too.
    let mut texts = vec![request.to_owned()];
    let mut runs = Vec::new();
    for (site, result) in sites.iter().zip(results) {
        if let Ok(AskOutcome::Done { args, output, .. }) = &result {
            texts.extend(args.values().filter_map(Value::as_str).map(str::to_owned));
            if let Output::Products(found) = output {
                products.extend(found.iter().cloned());
            }
        }
        let (outcome, error) = match result {
            Ok(outcome) => (Some(outcome), None),
            Err(e) => (None, Some(e.to_string())),
        };
        runs.push(SiteRun {
            site: site.name.clone(),
            outcome,
            error,
        });
    }

    let constraints = Constraints::from_request(request);
    let keywords = rank::keywords(texts.iter().map(String::as_str));
    let ranking = rank::rank(products, &constraints, &keywords);
    trace.record(Event::Ranked {
        constraints: ranking.constraints.clone(),
        keywords: ranking.keywords.clone(),
        kept: ranking
            .products
            .iter()
            .map(|r| RankedRef {
                source: r.product.source.clone(),
                id: r.product.id.clone(),
                name: r.product.name.clone(),
                price: r.product.price.clone(),
                url: r.product.url.clone(),
                score: r.score,
            })
            .collect(),
        excluded: ranking.excluded.clone(),
    });
    MultiOutcome {
        sites: runs,
        ranking,
    }
}

async fn ask_tab(
    tab: &Tab,
    site: &Site,
    request: &str,
    router: &Router,
    opts: &AskOptions,
    trace: &mut Trace,
) -> Result<AskOutcome, RunError> {
    let discovery = tab.wait_for_tools(opts.wait).await?;
    let specs: Vec<ToolSpec> = discovery.tools.iter().map(ToolSpec::from_tool).collect();
    trace.record(Event::ToolsDiscovered {
        url: discovery.url.clone(),
        origin: discovery.origin.clone(),
        tools: specs
            .iter()
            .map(|s| ToolRef {
                name: s.name.clone(),
                schema_hash: s.schema_hash.clone(),
            })
            .collect(),
        rejected: discovery.rejected.clone(),
    });
    if discovery.origin != site.origin {
        return Ok(block(
            trace,
            None,
            PolicyViolation::OriginMismatch {
                expected: site.origin.clone(),
                actual: discovery.origin,
            }
            .to_string(),
        ));
    }

    let mut allowed = Vec::new();
    for spec in specs {
        match site.check(&discovery.origin, &spec.name, &spec.schema_hash) {
            Ok(_) => allowed.push(spec),
            Err(violation) => trace.record(Event::ToolExcluded {
                tool: spec.name,
                reason: violation.to_string(),
            }),
        }
    }
    if allowed.is_empty() {
        return Ok(AskOutcome::NoTool);
    }

    let decision = router.route(request, &allowed).await?;
    for excluded in &decision.excluded {
        trace.record(Event::ToolExcluded {
            tool: excluded.name.clone(),
            reason: excluded.reasons.join("; "),
        });
    }
    trace.record(Event::Routed {
        source: decision.source.clone(),
        tool: decision.tool.clone(),
        args: decision.args.clone(),
        validation_errors: decision.validation_errors.clone(),
        latency_ms: decision.latency_ms,
        usage: decision.usage,
        llm_calls: decision.llm_calls,
        jev_usage: decision.jev_usage,
        jev_calls: decision.jev_calls,
        cached: decision.cached,
        hybrid: decision.hybrid.clone(),
    });
    let Some(tool) = decision.tool else {
        return Ok(AskOutcome::NoTool);
    };
    if !decision.validation_errors.is_empty() {
        return Ok(AskOutcome::Invalid {
            tool,
            errors: decision.validation_errors,
        });
    }

    // The page may have replaced the tool since discovery: check again.
    let current = tab.discover().await?;
    let Some(current_tool) = current.tool(&tool) else {
        let reason = format!("tool `{tool}` is no longer registered");
        return Ok(block(trace, Some(tool), reason));
    };
    let hash = schema_hash(&current_tool.input_schema);
    let capability = match site.check(&current.origin, &tool, &hash) {
        Ok(capability) => capability,
        Err(violation) => return Ok(block(trace, Some(tool), violation.to_string())),
    };
    trace.record(Event::PolicyChecked {
        tool: tool.clone(),
        schema_hash: hash,
    });

    let outcome = tab
        .execute(
            &tool,
            &Value::Object(decision.args.clone()),
            opts.call_timeout,
        )
        .await?;
    let raw = outcome.output.as_deref();
    trace.record(Event::Executed {
        tool: tool.clone(),
        args: decision.args.clone(),
        elapsed_ms: outcome.elapsed_ms,
        navigated: outcome.navigated(),
        url_after: outcome.url_after.clone(),
        output_bytes: raw.map_or(0, str::len),
        output_preview: raw.map(|r| trace::preview(r, trace::PREVIEW_CHARS)),
    });

    let source = Source {
        name: &site.name,
        currency: &site.currency,
        url: &site.url,
    };
    let output = adapters::adapt(capability.adapter.as_deref(), &source, raw)?;
    trace.record(Event::Adapted {
        adapter: capability.adapter.clone(),
        kind: output.kind().to_owned(),
        products: match &output {
            Output::Products(products) => products.len(),
            _ => 0,
        },
    });
    Ok(AskOutcome::Done {
        tool,
        args: decision.args,
        output,
    })
}

fn block(trace: &mut Trace, tool: Option<String>, reason: String) -> AskOutcome {
    trace.record(Event::Blocked {
        tool: tool.clone(),
        reason: reason.clone(),
    });
    AskOutcome::Blocked { tool, reason }
}
