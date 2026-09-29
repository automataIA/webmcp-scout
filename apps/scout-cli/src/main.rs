use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use scout::adapters::Output;
use scout::browser::{BrowserOptions, ChromeBrowser, Tab, TabOptions};
use scout::eval::{self, compare, dataset::Dataset, report, score::ScenarioResult};
use scout::jev::{JevClient, JevConfig, PINNED_MODEL};
use scout::llm::{CacheMode, LlmClient, LlmConfig};
use scout::local::{LocalClient, LocalConfig};
use scout::policy::Sites;
use scout::rank::Ranking;
use scout::router::Router;
use scout::router::baseline::BaselineRouter;
use scout::router::hybrid::{Gate, HybridRouter, Mode};
use scout::run::{self, AskOptions, AskOutcome, SiteRun};
use scout::schema::schema_hash;
use scout::trace::Trace;
use tokio::signal::unix::{Signal, SignalKind, signal};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "scout",
    version,
    about = "Agentic browser runtime for WebMCP sites"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Answer a request on the configured sites at once: route, check the
    /// allowlist, execute, adapt and rank the products of every site.
    Ask(AskArgs),
    /// List the WebMCP tools registered by a page.
    Tools {
        url: String,
        #[command(flatten)]
        page: PageArgs,
        /// Print the full discovery as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Execute one WebMCP tool and print its output.
    Call {
        url: String,
        tool: String,
        /// Tool arguments as a JSON object.
        #[arg(default_value = "{}")]
        args: String,
        #[command(flatten)]
        page: PageArgs,
        /// Tool call timeout in seconds.
        #[arg(long, default_value_t = 30)]
        timeout: u64,
    },
    /// Offline evaluations.
    #[command(subcommand)]
    Eval(EvalCommand),
}

#[derive(Subcommand)]
enum EvalCommand {
    /// Route the labeled scenarios with one router and write a report.
    Routing(RoutingArgs),
    /// Route the scenarios with the baseline and the hybrid routers of one
    /// decider, calibrate the gates and write the paired comparison.
    Compare(CompareArgs),
    /// Write the fan-out question set of every scenario, with its labels, as
    /// JSONL (no model is called).
    Questions(QuestionsArgs),
}

#[derive(Args)]
struct QuestionsArgs {
    #[arg(long, default_value = "evals/scenarios")]
    scenarios: PathBuf,
    #[arg(long, default_value = "evals/toolsets")]
    toolsets: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct RoutingArgs {
    #[arg(long, value_enum, default_value_t = RouterKind::Baseline)]
    router: RouterKind,
    #[command(flatten)]
    sources: EvalSources,
    /// Report name: writes `<out>/<name>.md` and `<out>/<name>.jsonl`.
    #[arg(long, default_value = "routing-baseline-v0")]
    name: String,
    /// Hybrid routers escalate to the baseline when tool metadata reads as an
    /// instruction to the model (`scout ask` always does).
    #[arg(long)]
    screen_metadata: bool,
}

#[derive(Args)]
struct CompareArgs {
    /// Who answers the closed questions: Jev (hybrid and fanout) or the
    /// local decision model (local).
    #[arg(long, value_enum, default_value_t = DeciderKind::Jev)]
    decider: DeciderKind,
    #[command(flatten)]
    sources: EvalSources,
    /// Report name: writes `<out>/<name>.md` and `<out>/<name>-<router>.jsonl`.
    #[arg(long, default_value = "routing-m4")]
    name: String,
}

#[derive(Args)]
struct EvalSources {
    /// Serve every LLM, Jev and local model answer from the caches; no API key,
    /// server or network needed.
    #[arg(long)]
    replay: bool,
    #[arg(long, default_value = "evals/scenarios")]
    scenarios: PathBuf,
    #[arg(long, default_value = "evals/toolsets")]
    toolsets: PathBuf,
    /// Recorded LLM responses, keyed by request hash.
    #[arg(long, default_value = "evals/cache/llm")]
    cache: PathBuf,
    /// Recorded Jev responses, keyed by request hash.
    #[arg(long, default_value = "evals/cache/jev")]
    jev_cache: PathBuf,
    /// Recorded local model responses, keyed by run name and request hash.
    #[arg(long, default_value = "evals/cache/local")]
    local_cache: PathBuf,
    #[command(flatten)]
    jev_pacing: JevPacing,
    #[arg(long, default_value = "docs/evals")]
    out: PathBuf,
}

#[derive(Args)]
struct AskArgs {
    request: String,
    /// Site name from the sites file; repeat for several. All sites if omitted.
    #[arg(long)]
    site: Vec<String>,
    /// Hybrid routers are gated: below the thresholds, or when the decider
    /// fails (no TYPESAFE_API_KEY, billing, network), they escalate to the baseline.
    #[arg(long, value_enum, default_value_t = RouterKind::Fanout)]
    router: RouterKind,
    #[command(flatten)]
    jev_pacing: JevPacing,
    /// Site config with the capability allowlist.
    #[arg(long, default_value = "sites.toml")]
    sites: PathBuf,
    /// Trace output; defaults to `traces/<unix time>-<site>[+<site>...].jsonl`.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Show the browser window.
    #[arg(long)]
    headed: bool,
    /// Seconds to wait for the page to register tools.
    #[arg(long, default_value_t = 15)]
    wait: u64,
    /// Tool call timeout in seconds.
    #[arg(long, default_value_t = 30)]
    timeout: u64,
    /// Seconds for a whole site; a slower site is dropped from the results.
    #[arg(long, default_value_t = 90)]
    site_timeout: u64,
    /// Ranked products to print; `--json` prints them all.
    #[arg(long, default_value_t = 10)]
    top: usize,
    /// Print the outcome as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct JevPacing {
    /// Least seconds between two Jev API requests (cache hits never wait);
    /// default 15 for evals, 0 for `ask`.
    #[arg(long)]
    jev_interval: Option<u64>,
    /// Seconds between retries of a request refused with HTTP 402 (billing),
    /// until Jev accepts it again; 0 fails at once. Default 30 for evals;
    /// 0 for `ask`, which then falls back to the baseline.
    #[arg(long = "jev-402-retry")]
    jev_billing_retry: Option<u64>,
}

/// Default (interval, 402 retry) seconds. Evals send hundreds of requests
/// and wait out billing gaps; `ask` sends one per site and falls back.
const EVAL_PACING: (u64, u64) = (15, 30);
const ASK_PACING: (u64, u64) = (0, 0);

impl JevPacing {
    fn apply(&self, jev: JevClient, (interval, billing_retry): (u64, u64)) -> JevClient {
        let interval = self.jev_interval.unwrap_or(interval);
        let jev = jev.with_min_interval(Duration::from_secs(interval));
        match self.jev_billing_retry.unwrap_or(billing_retry) {
            0 => jev,
            secs => jev.with_billing_retry(Duration::from_secs(secs)),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RouterKind {
    /// One strict LLM call picks the tool and every argument.
    Baseline,
    /// Jev stage 1, then stage 2 for the chosen tool; LLM for free strings.
    Hybrid,
    /// Jev stages 1 and 2 of every tool in one request; LLM for free strings.
    Fanout,
    /// The fan-out questions answered by the local decision model
    /// (`local_router/serve.py`, run named by SCOUT_LOCAL_MODEL); LLM for free strings.
    Local,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeciderKind {
    Jev,
    Local,
}

/// `gate`: whether hybrid routers escalate below the thresholds or when the
/// decider fails (`ask`), or return their raw decision (evals, which gate
/// post hoc and keep decider errors). `screen`: whether
/// they escalate when tool metadata reads as an instruction.
fn build_router(
    kind: RouterKind,
    llm: LlmClient,
    jev: JevClient,
    local: LocalClient,
    gate: bool,
    screen: bool,
) -> Router {
    let baseline = BaselineRouter::new(llm);
    let router = match kind {
        RouterKind::Baseline => return Router::Baseline(baseline),
        RouterKind::Hybrid => HybridRouter::new(jev, baseline, Mode::Staged),
        RouterKind::Fanout => HybridRouter::new(jev, baseline, Mode::FanOut),
        RouterKind::Local => HybridRouter::local(local, baseline),
    };
    let router = if screen {
        router.with_metadata_screen()
    } else {
        router
    };
    Router::Hybrid(if gate {
        router.with_gate(Gate::default()).with_decider_fallback()
    } else {
        router
    })
}

#[derive(Args)]
struct PageArgs {
    /// Install the shim for sites using the legacy `navigator.modelContext` API.
    #[arg(long)]
    legacy_shim: bool,
    /// Show the browser window.
    #[arg(long)]
    headed: bool,
    /// Seconds to wait for the page to register tools.
    #[arg(long, default_value_t = 15)]
    wait: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn,chromiumoxide=error".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let page = match &cli.command {
        Command::Tools { page, .. } | Command::Call { page, .. } => page,
        Command::Eval(EvalCommand::Routing(args)) => return eval_routing(args).await,
        Command::Eval(EvalCommand::Compare(args)) => return eval_compare(args).await,
        Command::Eval(EvalCommand::Questions(args)) => return eval_questions(args),
        Command::Ask(args) => return ask(args).await,
    };
    let mut interrupts = Interrupts::catch()?;
    let browser = ChromeBrowser::launch(&BrowserOptions {
        executable: None,
        headed: page.headed,
    })
    .await?;
    // On Ctrl-C or SIGTERM the command is dropped and Chrome still shut down.
    let result = tokio::select! {
        result = run(&browser, &cli.command) => result,
        signal = interrupts.recv() => Err(anyhow!("interrupted by {signal}")),
    };
    browser.shutdown().await;
    result
}

/// Ctrl-C and SIGTERM, caught from before Chrome is launched so that no
/// signal can end the process with Chrome and its profile left behind. Chrome
/// runs in our process group, so a terminal Ctrl-C may reach it first:
/// shutting down still reaps it and removes the temporary profile.
struct Interrupts {
    int: Signal,
    term: Signal,
}

impl Interrupts {
    fn catch() -> Result<Self> {
        Ok(Self {
            int: signal(SignalKind::interrupt())?,
            term: signal(SignalKind::terminate())?,
        })
    }

    /// Resolves on the next signal, including one received earlier.
    async fn recv(&mut self) -> &'static str {
        tokio::select! {
            _ = self.int.recv() => "SIGINT",
            _ = self.term.recv() => "SIGTERM",
        }
    }
}

async fn run(browser: &ChromeBrowser, command: &Command) -> Result<()> {
    match command {
        Command::Tools { url, page, json } => {
            let tab = open(browser, url, page).await?;
            let discovery = tab.wait_for_tools(Duration::from_secs(page.wait)).await?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&discovery)?);
                return Ok(());
            }
            println!("{} tool(s) on {}", discovery.tools.len(), discovery.url);
            for tool in &discovery.tools {
                println!("\n{}", tool.name);
                if !tool.description.is_empty() {
                    println!("  {}", tool.description);
                }
                println!("  input: {}", tool.input_schema);
                println!("  schema_hash: {}", schema_hash(&tool.input_schema));
            }
            for rejected in &discovery.rejected {
                println!("\nrejected {}: {}", rejected.name, rejected.reason);
            }
        }
        Command::Call {
            url,
            tool,
            args,
            page,
            timeout,
        } => {
            let args: serde_json::Value =
                serde_json::from_str(args).context("tool arguments are not valid JSON")?;
            if !args.is_object() {
                bail!("tool arguments must be a JSON object");
            }
            let tab = open(browser, url, page).await?;
            let discovery = tab.wait_for_tools(Duration::from_secs(page.wait)).await?;
            if discovery.tool(tool).is_none() {
                bail!("tool `{tool}` is not registered on {}", discovery.url);
            }
            let outcome = tab
                .execute(tool, &args, Duration::from_secs(*timeout))
                .await?;
            eprintln!(
                "{} returned in {} ms{}",
                outcome.tool,
                outcome.elapsed_ms,
                if outcome.navigated() {
                    format!(", page navigated to {}", outcome.url_after)
                } else {
                    String::new()
                }
            );
            println!("{}", outcome.output.as_deref().unwrap_or("null"));
        }
        Command::Eval(_) | Command::Ask(_) => {
            unreachable!("handled before launching the browser")
        }
    }
    Ok(())
}

/// Clients for an eval run, all behind their record/replay caches.
fn eval_clients(sources: &EvalSources) -> Result<(LlmClient, JevClient, LocalClient)> {
    // A missing .env is fine: the variables may come from the environment.
    let _ = dotenvy::dotenv();
    let mode = if sources.replay {
        CacheMode::Replay
    } else {
        CacheMode::Record
    };
    let llm = LlmClient::new(LlmConfig::from_env())?.with_cache(&sources.cache, mode);
    // Evals pin the Jev model so that recorded answers stay comparable.
    let jev = JevClient::new(JevConfig {
        model: PINNED_MODEL.into(),
        ..JevConfig::from_env()
    })?
    .with_cache(&sources.jev_cache, mode);
    let local = LocalClient::new(LocalConfig::from_env())?.with_cache(&sources.local_cache, mode);
    Ok((llm, sources.jev_pacing.apply(jev, EVAL_PACING), local))
}

fn write_file(path: &Path, contents: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("cannot write {}", path.display()))
}

async fn eval_routing(args: &RoutingArgs) -> Result<()> {
    let sources = &args.sources;
    let dataset = Dataset::load(&sources.scenarios, &sources.toolsets)?;
    let (llm, jev, local) = eval_clients(sources)?;
    let mut meta = match args.router {
        RouterKind::Baseline => report::baseline_meta(&args.name, llm.model()),
        RouterKind::Local => report::local_meta(&args.name, llm.model(), local.model()),
        _ => report::hybrid_meta(&args.name, llm.model(), jev.model()),
    };
    if args.screen_metadata && args.router != RouterKind::Baseline {
        meta.router
            .push_str(", escalated to the baseline when tool metadata reads as an instruction");
    }
    let router = build_router(args.router, llm, jev, local, false, args.screen_metadata);
    let results = route_scenarios(&dataset, &router).await?;

    let md = sources.out.join(format!("{}.md", args.name));
    let jsonl = sources.out.join(format!("{}.jsonl", args.name));
    write_file(&md, &report::render_markdown(&meta, &results))?;
    write_file(&jsonl, &report::render_jsonl(&results))?;
    eprintln!("wrote {} and {}", md.display(), jsonl.display());
    Ok(())
}

async fn eval_compare(args: &CompareArgs) -> Result<()> {
    let sources = &args.sources;
    let dataset = Dataset::load(&sources.scenarios, &sources.toolsets)?;
    let mut runs = Vec::new();
    // One Jev client: its clones share the request pacing across runs.
    let (llm, jev, local) = eval_clients(sources)?;
    let llm_model = llm.model().to_owned();
    let (decider, kinds) = match args.decider {
        DeciderKind::Jev => (
            compare::Decider::Jev(jev.model().to_owned()),
            vec![RouterKind::Baseline, RouterKind::Hybrid, RouterKind::Fanout],
        ),
        DeciderKind::Local => (
            compare::Decider::Local(local.model().to_owned()),
            vec![RouterKind::Baseline, RouterKind::Local],
        ),
    };
    for kind in kinds {
        let router = build_router(kind, llm.clone(), jev.clone(), local.clone(), false, false);
        eprintln!("== {}", router.name());
        let results = route_scenarios(&dataset, &router).await?;
        let jsonl = sources
            .out
            .join(format!("{}-{}.jsonl", args.name, router.name()));
        write_file(&jsonl, &report::render_jsonl(&results))?;
        eprintln!("wrote {}", jsonl.display());
        runs.push(compare::Variant {
            name: router.name().to_owned(),
            results,
        });
    }
    let baseline = runs.remove(0);
    let hybrids = runs
        .into_iter()
        .map(|v| {
            let gate = compare::calibrate(&v.results);
            (v, gate)
        })
        .collect();
    let comparison = compare::Comparison {
        llm_model,
        decider,
        baseline,
        hybrids,
    };
    let md = sources.out.join(format!("{}.md", args.name));
    write_file(&md, &comparison.render_markdown())?;
    eprintln!("wrote {}", md.display());
    Ok(())
}

fn eval_questions(args: &QuestionsArgs) -> Result<()> {
    let dataset = Dataset::load(&args.scenarios, &args.toolsets)?;
    let mut out = String::new();
    for record in eval::question_records(&dataset) {
        out.push_str(&serde_json::to_string(&record)?);
        out.push('\n');
    }
    write_file(&args.out, &out)?;
    eprintln!(
        "wrote {} question sets to {}",
        dataset.scenarios.len(),
        args.out.display()
    );
    Ok(())
}

/// Routes every scenario, printing one progress line per result.
async fn route_scenarios(dataset: &Dataset, router: &Router) -> Result<Vec<ScenarioResult>> {
    let total = dataset.scenarios.len();
    let mut done = 0;
    let results = eval::run_routing(dataset, router, |r| {
        done += 1;
        eprintln!(
            "[{done}/{total}] {} {} -> {}{}",
            if r.exact { "ok  " } else { "MISS" },
            r.id,
            r.predicted_tool.as_deref().unwrap_or("none"),
            r.error
                .as_deref()
                .map(|e| format!(" (error: {e})"))
                .unwrap_or_default()
        );
    })
    .await?;
    Ok(results)
}

async fn ask(args: &AskArgs) -> Result<()> {
    // A missing .env is fine: the variables may come from the environment.
    let _ = dotenvy::dotenv();
    let sites = Sites::load(&args.sites)?;
    let names: Vec<&str> = if args.site.is_empty() {
        sites.names().collect()
    } else {
        args.site.iter().map(String::as_str).collect()
    };
    let selected = names
        .iter()
        .map(|name| sites.get(name))
        .collect::<Result<Vec<_>, _>>()?;
    let router = build_router(
        args.router,
        LlmClient::new(LlmConfig::from_env())?,
        args.jev_pacing
            .apply(JevClient::new(JevConfig::from_env())?, ASK_PACING),
        LocalClient::new(LocalConfig::from_env())?,
        true,
        true,
    );
    let opts = AskOptions {
        wait: Duration::from_secs(args.wait),
        call_timeout: Duration::from_secs(args.timeout),
        site_timeout: Duration::from_secs(args.site_timeout),
    };
    let mut interrupts = Interrupts::catch()?;
    let browser = ChromeBrowser::launch(&BrowserOptions {
        executable: None,
        headed: args.headed,
    })
    .await?;
    let mut trace = Trace::new();
    let outcome = tokio::select! {
        outcome = run::ask_sites(
            &browser,
            &selected,
            &args.request,
            &router,
            &opts,
            &mut trace,
        ) => Ok(outcome),
        signal = interrupts.recv() => Err(signal),
    };
    browser.shutdown().await;

    let path = args.trace.clone().unwrap_or_else(|| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        PathBuf::from(format!("traces/{now}-{}.jsonl", names.join("+")))
    });
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    std::fs::write(&path, trace.to_jsonl())
        .with_context(|| format!("cannot write {}", path.display()))?;
    eprintln!("trace: {}", path.display());
    // The trace of an interrupted run keeps the steps done so far.
    let outcome = outcome.map_err(|signal| anyhow!("interrupted by {signal}"))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
    } else {
        for site in &outcome.sites {
            print_site(site);
        }
        print_ranking(&outcome.ranking, args.top);
    }
    Ok(())
}

fn print_site(site: &SiteRun) {
    print!("[{}] ", site.site);
    let Some(outcome) = &site.outcome else {
        println!("error: {}", site.error.as_deref().unwrap_or("unknown"));
        return;
    };
    match outcome {
        AskOutcome::NoTool => println!("No allowlisted tool fits the request."),
        AskOutcome::Invalid { tool, errors } => {
            println!("Not executed: arguments for `{tool}` are invalid:");
            for error in errors {
                println!("  - {error}");
            }
        }
        AskOutcome::Blocked { tool, reason } => match tool {
            Some(tool) => println!("Blocked `{tool}`: {reason}"),
            None => println!("Blocked: {reason}"),
        },
        AskOutcome::Done { tool, args, output } => {
            println!("{tool} {}", serde_json::Value::Object(args.clone()));
            match output {
                Output::Products(products) => println!("  {} product(s)", products.len()),
                Output::Json(value) => println!(
                    "{}",
                    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
                ),
                Output::Status(text) => println!("  {text}"),
                Output::Empty => println!("  (no output)"),
            }
        }
    }
}

fn print_ranking(ranking: &Ranking, top: usize) {
    let c = &ranking.constraints;
    let mut filters = vec!["in stock".to_owned()];
    if let Some(max) = c.max_price {
        filters.push(format!(
            "price <= {max} {}",
            c.currency.as_deref().unwrap_or("(any currency)")
        ));
    }
    if let Some(size) = &c.size {
        filters.push(format!("size {size}"));
    }
    println!(
        "\n{} product(s) ranked, {} excluded ({})",
        ranking.products.len(),
        ranking.excluded.len(),
        filters.join(", ")
    );
    for (i, r) in ranking.products.iter().take(top).enumerate() {
        let p = &r.product;
        let price = p.price.as_ref().map_or_else(
            || "–".to_owned(),
            |m| format!("{:.2} {}", m.amount, m.currency),
        );
        let stock = p
            .availability
            .map_or_else(|| "–".to_owned(), |n| n.to_string());
        println!(
            "{:>3}. {:.2}  {}  {price}  stock {stock}  [{} {}]",
            i + 1,
            r.score,
            p.name,
            p.source,
            p.id
        );
    }
    if ranking.products.len() > top {
        println!("  ... {} more", ranking.products.len() - top);
    }
    for e in &ranking.excluded {
        println!("  excluded [{} {}]: {}", e.source, e.id, e.reason);
    }
}

async fn open(browser: &ChromeBrowser, url: &str, page: &PageArgs) -> Result<Tab> {
    let opts = TabOptions {
        legacy_shim: page.legacy_shim,
    };
    browser
        .open(url, &opts)
        .await
        .with_context(|| format!("failed to open {url}"))
}
