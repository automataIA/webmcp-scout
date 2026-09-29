//! `run::ask_site` end to end: real Chrome on the `alpha` fixture, with the
//! LLM served by wiremock so every run is offline and deterministic.
//! The ignored test drives the live sport-shop with the real Groq model:
//! `SCOUT_LIVE=1 cargo test -p scout --test ask -- --ignored`.

mod common;

use std::time::Duration;

use common::FixtureServer;
use scout::adapters::Output;
use scout::browser::{BrowserOptions, ChromeBrowser};
use scout::llm::{LlmClient, LlmConfig};
use scout::policy::{Capability, CapabilityClass, Site, SiteConfig, Sites};
use scout::router::Router;
use scout::router::baseline::BaselineRouter;
use scout::run::{AskOptions, AskOutcome, ask_site};
use scout::trace::{Event, Trace};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `schema_hash` of the alpha `search_products` input schema (`scout tools`).
const SEARCH_HASH: &str = "sha256:9f850f72599a0fd26fce73d85a27953ecf31651fbfc1b35ad0a561189103ccf9";

const OPTS: AskOptions = AskOptions {
    wait: Duration::from_secs(10),
    call_timeout: Duration::from_secs(10),
    site_timeout: Duration::from_secs(60),
};

fn alpha(server: &FixtureServer, schema_hash: &str) -> Site {
    Site::new(
        "alpha",
        SiteConfig {
            url: server.url("alpha/"),
            legacy_shim: false,
            currency: "EUR".into(),
            capabilities: vec![Capability {
                tool: "search_products".into(),
                schema_hash: schema_hash.into(),
                class: CapabilityClass::ReadOnly,
                adapter: Some("alpha.search_products.v1".into()),
            }],
        },
    )
    .unwrap()
}

/// A mock LLM that answers every request with `call` as the routing output.
async fn llm(call: Value, expected_requests: u64) -> MockServer {
    let server = MockServer::start().await;
    let content = json!({ "call": call }).to_string();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "finish_reason": "stop",
                "message": { "role": "assistant", "content": content }
            }],
            "usage": { "prompt_tokens": 100, "completion_tokens": 20 }
        })))
        .expect(expected_requests)
        .mount(&server)
        .await;
    server
}

fn router(server: &MockServer) -> Router {
    Router::Baseline(BaselineRouter::new(
        LlmClient::new(LlmConfig {
            base_url: server.uri(),
            api_key: Some("test-key".into()),
            model: "test-model".into(),
            reasoning_effort: None,
        })
        .unwrap(),
    ))
}

async fn ask(site: &Site, router: &Router) -> (AskOutcome, Trace) {
    let browser = ChromeBrowser::launch(&BrowserOptions::default())
        .await
        .expect("launch Chrome");
    let mut trace = Trace::new();
    let outcome = ask_site(
        &browser,
        site,
        "soccer boots under 150 euros",
        router,
        &OPTS,
        &mut trace,
    )
    .await;
    browser.shutdown().await;
    (outcome.expect("ask_site"), trace)
}

fn events(trace: &Trace) -> Vec<&Event> {
    trace.records().iter().map(|r| &r.event).collect()
}

fn executed(trace: &Trace) -> bool {
    events(trace)
        .iter()
        .any(|e| matches!(e, Event::Executed { .. }))
}

#[tokio::test]
async fn allowlisted_tool_runs_and_its_output_is_adapted() {
    let fixtures = FixtureServer::start().await;
    let llm = llm(
        json!({
            "tool": "search_products",
            "arguments": { "query": "soccer boots", "category": null, "max_price": 150 }
        }),
        1,
    )
    .await;
    let (outcome, trace) = ask(&alpha(&fixtures, SEARCH_HASH), &router(&llm)).await;

    let AskOutcome::Done { tool, args, output } = outcome else {
        panic!("expected done, got {outcome:?}");
    };
    assert_eq!(tool, "search_products");
    assert_eq!(
        Value::Object(args),
        json!({ "query": "soccer boots", "max_price": 150 })
    );
    let Output::Products(products) = output else {
        panic!("expected products, got {output:?}");
    };
    let ids: Vec<&str> = products.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["A-100", "A-102"]);
    assert_eq!(products[0].price.as_ref().unwrap().currency, "EUR");

    // The model never saw the tool outside the allowlist.
    let requests = llm.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("search_products") && !body.contains("get_product"));

    let events = events(&trace);
    assert!(events.iter().any(|e| matches!(
        e,
        Event::ToolExcluded { tool, .. } if tool == "get_product"
    )));
    let kinds: Vec<String> = trace
        .records()
        .iter()
        .map(|r| {
            serde_json::to_value(r).unwrap()["event"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "run_started",
            "tools_discovered",
            "tool_excluded",
            "routed",
            "policy_checked",
            "executed",
            "adapted",
            "run_finished"
        ]
    );
}

#[tokio::test]
async fn changed_schema_hash_blocks_the_tool_before_routing() {
    let fixtures = FixtureServer::start().await;
    let llm = llm(json!({ "tool": "none", "arguments": null }), 0).await;
    let stale = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let (outcome, trace) = ask(&alpha(&fixtures, stale), &router(&llm)).await;

    assert_eq!(outcome, AskOutcome::NoTool);
    assert!(!executed(&trace));
    assert!(events(&trace).iter().any(|e| matches!(
        e,
        Event::ToolExcluded { tool, reason } if tool == "search_products" && reason.contains("changed its input schema")
    )));
}

#[tokio::test]
async fn no_tool_and_invalid_arguments_execute_nothing() {
    let fixtures = FixtureServer::start().await;
    let site = alpha(&fixtures, SEARCH_HASH);

    let none = llm(json!({ "tool": "none", "arguments": null }), 1).await;
    let (outcome, trace) = ask(&site, &router(&none)).await;
    assert_eq!(outcome, AskOutcome::NoTool);
    assert!(!executed(&trace));

    let invalid = llm(
        json!({
            "tool": "search_products",
            "arguments": { "query": "boots", "category": null, "max_price": -5 }
        }),
        1,
    )
    .await;
    let (outcome, trace) = ask(&site, &router(&invalid)).await;
    assert!(
        matches!(&outcome, AskOutcome::Invalid { tool, .. } if tool == "search_products"),
        "{outcome:?}"
    );
    assert!(!executed(&trace));
}

#[tokio::test]
#[ignore = "live: needs network, Chrome and a Groq key"]
async fn live_sport_shop_search() {
    let _ = dotenvy::dotenv();
    let sites =
        Sites::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sites.toml"))
            .unwrap();
    let site = sites.get("sport").unwrap();
    let router = Router::Baseline(BaselineRouter::new(
        LlmClient::new(LlmConfig::from_env()).unwrap(),
    ));
    let browser = ChromeBrowser::launch(&BrowserOptions::default())
        .await
        .unwrap();
    let mut trace = Trace::new();
    let opts = AskOptions {
        wait: Duration::from_secs(30),
        call_timeout: Duration::from_secs(20),
        site_timeout: Duration::from_secs(90),
    };
    let outcome = ask_site(&browser, site, "soccer boots", &router, &opts, &mut trace).await;
    browser.shutdown().await;

    let AskOutcome::Done { tool, output, .. } = outcome.unwrap() else {
        panic!("expected done: {}", trace.to_jsonl());
    };
    assert_eq!(tool, "search_product");
    let Output::Products(products) = output else {
        panic!("expected products");
    };
    assert!(!products.is_empty());
    assert!(products.iter().all(|p| p.source == "sport"));
}
