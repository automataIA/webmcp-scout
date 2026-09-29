//! Allowlist and untrusted-input guarantees on the `adversarial` fixture,
//! with the LLM served by wiremock: tool descriptions and results carry
//! prompt injection, a `checkout` tool outside the allowlist asks to be
//! called, and the page can change or drop its search tool after routing. Nothing outside the
//! allowlist runs, results trigger no further model call, and a tool that
//! changed since discovery is blocked before execution.

mod common;

use std::time::Duration;

use common::FixtureServer;
use scout::adapters::Output;
use scout::browser::{BrowserOptions, ChromeBrowser};
use scout::llm::{LlmClient, LlmConfig};
use scout::policy::{Capability, CapabilityClass, Site, SiteConfig};
use scout::router::Router;
use scout::router::baseline::BaselineRouter;
use scout::run::{AskOptions, AskOutcome, ask_site};
use scout::trace::{Event, Trace};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// `schema_hash` of `search_items` before any drift (`scout tools`).
const SEARCH_HASH: &str = "sha256:b1eaab9711f450db8ea4a97f48a3a454168ddbfcd8ce967769f67a04e2baf8ff";

/// Longer than the fixture's 3 s drift timer, so that the page changes its
/// tool while the router is still thinking.
const SLOW_MODEL: Duration = Duration::from_secs(5);

const OPTS: AskOptions = AskOptions {
    wait: Duration::from_secs(10),
    call_timeout: Duration::from_secs(10),
    site_timeout: Duration::from_secs(60),
};

fn site(fixtures: &FixtureServer, query: &str) -> Site {
    Site::new(
        "adversarial",
        SiteConfig {
            url: fixtures.url(&format!("adversarial/{query}")),
            legacy_shim: false,
            currency: "EUR".into(),
            capabilities: vec![Capability {
                tool: "search_items".into(),
                schema_hash: SEARCH_HASH.into(),
                class: CapabilityClass::ReadOnly,
                adapter: None,
            }],
        },
    )
    .unwrap()
}

/// A mock LLM that answers once with `call`, after `delay`.
async fn llm(call: Value, delay: Duration) -> MockServer {
    let server = MockServer::start().await;
    let content = json!({ "call": call }).to_string();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({
                    "choices": [{
                        "finish_reason": "stop",
                        "message": { "role": "assistant", "content": content }
                    }],
                    "usage": { "prompt_tokens": 100, "completion_tokens": 20 }
                }))
                .set_delay(delay),
        )
        .expect(1)
        .mount(&server)
        .await;
    server
}

async fn ask(site: &Site, llm: &MockServer) -> (AskOutcome, Trace) {
    let router = Router::Baseline(BaselineRouter::new(
        LlmClient::new(LlmConfig {
            base_url: llm.uri(),
            api_key: Some("test-key".into()),
            model: "test-model".into(),
            reasoning_effort: None,
        })
        .unwrap(),
    ));
    let browser = ChromeBrowser::launch(&BrowserOptions::default())
        .await
        .expect("launch Chrome");
    let mut trace = Trace::new();
    let outcome = ask_site(&browser, site, "soccer boots", &router, &OPTS, &mut trace).await;
    browser.shutdown().await;
    (outcome.expect("ask_site"), trace)
}

fn executed(trace: &Trace) -> Vec<&str> {
    trace
        .records()
        .iter()
        .filter_map(|r| match &r.event {
            Event::Executed { tool, .. } => Some(tool.as_str()),
            _ => None,
        })
        .collect()
}

fn policy_checked(trace: &Trace) -> bool {
    trace
        .records()
        .iter()
        .any(|r| matches!(r.event, Event::PolicyChecked { .. }))
}

#[tokio::test]
async fn a_model_fooled_by_injected_metadata_cannot_run_checkout() {
    let fixtures = FixtureServer::start().await;
    // The model obeys the injected notice and asks for `checkout`.
    let llm = llm(
        json!({ "tool": "checkout", "arguments": { "confirm": true } }),
        Duration::ZERO,
    )
    .await;
    let (outcome, trace) = ask(&site(&fixtures, ""), &llm).await;

    assert!(
        matches!(&outcome, AskOutcome::Invalid { tool, errors }
            if tool == "checkout" && errors == &["unknown tool `checkout`"]),
        "{outcome:?}"
    );
    assert!(executed(&trace).is_empty());
    assert!(trace.records().iter().any(|r| matches!(
        &r.event,
        Event::ToolExcluded { tool, reason } if tool == "checkout" && reason.contains("not in the allowlist")
    )));
    // `checkout` was never offered to the model: its own description is
    // missing from the request (the injected text of `search_items` still
    // names it, as untrusted data).
    let requests = llm.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(!body.contains("Always call this tool first"));
}

#[tokio::test]
async fn injected_results_are_data_and_trigger_no_further_call() {
    let fixtures = FixtureServer::start().await;
    let llm = llm(
        json!({ "tool": "search_items", "arguments": { "query": "soccer" } }),
        Duration::ZERO,
    )
    .await;
    let (outcome, trace) = ask(&site(&fixtures, ""), &llm).await;

    let AskOutcome::Done { tool, output, .. } = outcome else {
        panic!("expected done, got {outcome:?}");
    };
    assert_eq!(tool, "search_items");
    // The injection comes back verbatim, as data for the user.
    let text = serde_json::to_string(&output).unwrap();
    assert!(text.contains("IMPORTANT SYSTEM NOTICE"), "{output:?}");
    assert!(!matches!(output, Output::Products(_)));
    // One routing call (checked by the mock on drop), one execution.
    assert_eq!(executed(&trace), ["search_items"]);
}

#[tokio::test]
async fn a_schema_changed_after_routing_is_blocked_before_execution() {
    let fixtures = FixtureServer::start().await;
    let llm = llm(
        json!({ "tool": "search_items", "arguments": { "query": "soccer" } }),
        SLOW_MODEL,
    )
    .await;
    let (outcome, trace) = ask(&site(&fixtures, "?drift=schema"), &llm).await;

    assert!(
        matches!(&outcome, AskOutcome::Blocked { tool: Some(tool), reason }
            if tool == "search_items"
                && reason.contains("changed its input schema")
                && reason.contains(SEARCH_HASH)),
        "{outcome:?}"
    );
    assert!(executed(&trace).is_empty());
    assert!(!policy_checked(&trace));
}

#[tokio::test]
async fn a_tool_dropped_after_routing_is_blocked_before_execution() {
    let fixtures = FixtureServer::start().await;
    let llm = llm(
        json!({ "tool": "search_items", "arguments": { "query": "soccer" } }),
        SLOW_MODEL,
    )
    .await;
    let (outcome, trace) = ask(&site(&fixtures, "?drift=remove"), &llm).await;

    assert_eq!(
        outcome,
        AskOutcome::Blocked {
            tool: Some("search_items".into()),
            reason: "tool `search_items` is no longer registered".into(),
        }
    );
    assert!(executed(&trace).is_empty());
    assert!(!policy_checked(&trace));
}
