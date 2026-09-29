//! `run::ask_sites` end to end on the `alpha` and `beta` fixtures in one real
//! Chrome, with the LLM served by wiremock: concurrent sites, a site that
//! times out without spoiling the others, and one ranking over both shops.

mod common;

use std::time::Duration;

use common::FixtureServer;
use scout::llm::{LlmClient, LlmConfig};
use scout::policy::{Capability, CapabilityClass, Site, SiteConfig};
use scout::rank::Constraints;
use scout::router::Router;
use scout::router::baseline::BaselineRouter;
use scout::run::{AskOptions, AskOutcome, ask_sites};
use scout::trace::{Event, Trace};
use serde_json::{Value, json};
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `schema_hash` values from `scout tools` on the fixtures.
const ALPHA_HASH: &str = "sha256:9f850f72599a0fd26fce73d85a27953ecf31651fbfc1b35ad0a561189103ccf9";
const BETA_HASH: &str = "sha256:60234ee2d10d52a1de741f2f554b9b16837d303f3cd57de566a13554365c6994";

const REQUEST: &str = "soccer boots under 150 euros, size 42";

fn site(name: &str, url: String, tool: &str, hash: &str, adapter: Option<&str>) -> Site {
    Site::new(
        name,
        SiteConfig {
            url,
            legacy_shim: false,
            currency: "EUR".into(),
            capabilities: vec![Capability {
                tool: tool.into(),
                schema_hash: hash.into(),
                class: CapabilityClass::ReadOnly,
                adapter: adapter.map(Into::into),
            }],
        },
    )
    .unwrap()
}

/// Answers the routing request that lists `tool` with `call`.
async fn route(server: &MockServer, tool: &str, call: Value) {
    let content = json!({ "call": call }).to_string();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(tool))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "finish_reason": "stop",
                "message": { "role": "assistant", "content": content }
            }],
            "usage": { "prompt_tokens": 100, "completion_tokens": 20 }
        })))
        .expect(1)
        .mount(server)
        .await;
}

#[tokio::test]
async fn two_shops_are_ranked_together_and_a_slow_site_is_dropped() {
    let fixtures = FixtureServer::start().await;
    let llm = MockServer::start().await;
    route(
        &llm,
        "search_products",
        json!({
            "tool": "search_products",
            "arguments": { "query": "soccer boots", "category": null, "max_price": 150 }
        }),
    )
    .await;
    route(
        &llm,
        "find_shoes",
        json!({ "tool": "find_shoes", "arguments": { "keywords": "soccer boots" } }),
    )
    .await;
    let router = Router::Baseline(BaselineRouter::new(
        LlmClient::new(LlmConfig {
            base_url: llm.uri(),
            api_key: Some("test-key".into()),
            model: "test-model".into(),
            reasoning_effort: None,
        })
        .unwrap(),
    ));

    let alpha = site(
        "alpha",
        fixtures.url("alpha/"),
        "search_products",
        ALPHA_HASH,
        Some("alpha.search_products.v1"),
    );
    let beta = site(
        "beta",
        fixtures.url("beta/"),
        "find_shoes",
        BETA_HASH,
        Some("beta.find_shoes.v1"),
    );
    // Without the legacy shim this page never registers a tool, so it waits
    // until the site budget runs out.
    let slow = site(
        "slow",
        fixtures.url("legacy/"),
        "list_items",
        ALPHA_HASH,
        None,
    );
    let opts = AskOptions {
        wait: Duration::from_secs(60),
        call_timeout: Duration::from_secs(10),
        site_timeout: Duration::from_secs(8),
    };

    let browser = scout::browser::ChromeBrowser::launch(&Default::default())
        .await
        .expect("launch Chrome");
    let mut trace = Trace::new();
    let outcome = ask_sites(
        &browser,
        &[&alpha, &beta, &slow],
        REQUEST,
        &router,
        &opts,
        &mut trace,
    )
    .await;
    browser.shutdown().await;

    let status: Vec<(&str, Option<&str>, Option<&str>)> = outcome
        .sites
        .iter()
        .map(|s| {
            (
                s.site.as_str(),
                s.outcome.as_ref().map(AskOutcome::status),
                s.error.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        status,
        [
            ("alpha", Some("done"), None),
            ("beta", Some("done"), None),
            ("slow", None, Some("site timed out after 8s")),
        ]
    );

    let ranking = &outcome.ranking;
    assert_eq!(
        ranking.constraints,
        Constraints {
            max_price: Some(150.0),
            currency: Some("EUR".into()),
            size: Some("42".into()),
        }
    );
    assert_eq!(ranking.keywords, ["soccer", "boot"]);
    let kept: Vec<(&str, &str, f64)> = ranking
        .products
        .iter()
        .map(|r| (r.product.source.as_str(), r.product.id.as_str(), r.score))
        .collect();
    assert_eq!(
        kept,
        [
            ("alpha", "A-100", 1.0),
            ("beta", "B-4", 0.5),
            ("beta", "B-1", 0.5),
        ]
    );
    let excluded: Vec<(&str, &str)> = ranking
        .excluded
        .iter()
        .map(|e| (e.id.as_str(), e.reason.as_str()))
        .collect();
    assert_eq!(
        excluded,
        [
            ("A-102", "out of stock"),
            ("B-2", "size 42 not listed"),
            ("B-3", "out of stock"),
        ]
    );

    // One trace, every record tagged with its site, the ranking last.
    let records = trace.records();
    for name in ["alpha", "beta", "slow"] {
        assert!(
            records
                .iter()
                .any(|r| r.site.as_deref() == Some(name)
                    && matches!(r.event, Event::RunFinished { .. })),
            "no run_finished for {name}"
        );
    }
    let last = records.last().unwrap();
    assert!(last.site.is_none() && matches!(last.event, Event::Ranked { .. }));
    assert!(records.windows(2).all(|w| w[0].t_ms <= w[1].t_ms));
}

#[tokio::test]
#[ignore = "live: needs network, Chrome and a Groq key"]
async fn live_sport_and_shoe_store() {
    let _ = dotenvy::dotenv();
    let sites = scout::policy::Sites::load(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sites.toml"),
    )
    .unwrap();
    let selected = [sites.get("sport").unwrap(), sites.get("shoes").unwrap()];
    let router = Router::Baseline(BaselineRouter::new(
        LlmClient::new(LlmConfig::from_env()).unwrap(),
    ));
    let opts = AskOptions {
        wait: Duration::from_secs(30),
        ..AskOptions::default()
    };
    let browser = scout::browser::ChromeBrowser::launch(&Default::default())
        .await
        .unwrap();
    let mut trace = Trace::new();
    let outcome = ask_sites(
        &browser,
        &selected,
        "soccer shoes under 150 dollars",
        &router,
        &opts,
        &mut trace,
    )
    .await;
    browser.shutdown().await;

    for site in &outcome.sites {
        assert_eq!(
            site.outcome.as_ref().map(AskOutcome::status),
            Some("done"),
            "{}: {:?}\n{}",
            site.site,
            site.error,
            trace.to_jsonl()
        );
    }
    let ranking = &outcome.ranking;
    assert_eq!(ranking.constraints.max_price, Some(150.0));
    for source in ["sport", "shoes"] {
        assert!(ranking.products.iter().any(|r| r.product.source == source));
    }
    assert!(ranking.products.iter().all(|r| {
        let price = r.product.price.as_ref().unwrap();
        price.amount <= 150.0 && price.currency == "USD" && r.product.availability != Some(0)
    }));
}
