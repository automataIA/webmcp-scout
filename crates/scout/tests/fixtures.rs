//! Browser runtime against the local fixture sites. Requires a Chrome with
//! WebMCP (see `resolve_chrome`).

mod common;

use std::time::Duration;

use common::FixtureServer;
use scout::browser::{ArgsEncoding, BrowserError, BrowserOptions, ChromeBrowser, Tab, TabOptions};
use serde_json::{Value, json};

const WAIT: Duration = Duration::from_secs(10);
const CALL: Duration = Duration::from_secs(10);

struct Env {
    server: FixtureServer,
    browser: ChromeBrowser,
}

impl Env {
    async fn start() -> Self {
        let server = FixtureServer::start().await;
        let browser = ChromeBrowser::launch(&BrowserOptions::default())
            .await
            .expect("launch Chrome");
        Self { server, browser }
    }

    async fn open(&self, site: &str, legacy_shim: bool) -> Tab {
        self.browser
            .open(&self.server.url(site), &TabOptions { legacy_shim })
            .await
            .expect("open fixture")
    }

    async fn stop(self) {
        self.browser.shutdown().await;
    }
}

fn parse(output: Option<String>) -> Value {
    serde_json::from_str(&output.expect("tool output")).expect("JSON tool output")
}

#[tokio::test]
async fn discovers_echo_tools_with_parsed_schemas() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    let discovery = tab.wait_for_tools(WAIT).await.unwrap();

    let mut names: Vec<_> = discovery.tools.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["echo", "echo_object", "fail", "navigate_away", "set_hash"]
    );
    assert!(discovery.rejected.is_empty());
    let echo = discovery.tool("echo").unwrap();
    assert_eq!(echo.input_schema["required"], json!(["message"]));
    assert_eq!(echo.input_schema["properties"]["count"]["type"], "integer");
    assert_eq!(echo.origin, discovery.origin);
    env.stop().await;
}

/// Spike: which argument encoding `executeTool` accepts. Run only against the
/// idempotent echo fixture.
#[tokio::test]
async fn execute_tool_requires_json_string_arguments() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    tab.wait_for_tools(WAIT).await.unwrap();
    let args = json!({ "message": "hi", "count": 2 });

    let as_string = tab
        .execute_with_encoding("echo", &args, ArgsEncoding::JsonString, CALL)
        .await
        .unwrap();
    // Chrome parses the string and hands the tool a plain object.
    assert_eq!(
        parse(as_string.output),
        json!({ "type": "object", "received": args })
    );

    let as_object = tab
        .execute_with_encoding("echo", &args, ArgsEncoding::Object, CALL)
        .await;
    assert!(
        matches!(&as_object, Err(BrowserError::ToolFailed { message, .. }) if message.contains("parse input")),
        "Chrome now accepts object arguments, revisit DEFAULT_ARGS_ENCODING: {as_object:?}"
    );
    env.stop().await;
}

#[tokio::test]
async fn non_string_results_are_json_encoded() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    tab.wait_for_tools(WAIT).await.unwrap();

    let outcome = tab.execute("echo_object", &json!({}), CALL).await.unwrap();
    assert!(!outcome.navigated());
    assert_eq!(
        parse(outcome.output),
        json!({ "ok": true, "items": [1, 2] })
    );
    env.stop().await;
}

#[tokio::test]
async fn same_document_navigation_is_reported() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    tab.wait_for_tools(WAIT).await.unwrap();

    let outcome = tab
        .execute("set_hash", &json!({ "hash": "x" }), CALL)
        .await
        .unwrap();
    assert!(outcome.navigated());
    assert!(
        outcome.url_after.ends_with("/echo/#x"),
        "{}",
        outcome.url_after
    );
    env.stop().await;
}

/// Contract: a tool that navigates the whole page loses its JS context; the
/// caller must rediscover on the new document.
#[tokio::test]
async fn full_navigation_during_call_destroys_context() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    tab.wait_for_tools(WAIT).await.unwrap();

    let result = tab.execute("navigate_away", &json!({}), CALL).await;
    assert!(
        matches!(result, Err(BrowserError::ContextDestroyed)),
        "{result:?}"
    );
    // The new document has no tools: rediscovery retries while the context is
    // being replaced and returns the empty discovery at the deadline.
    let after = tab.wait_for_tools(Duration::from_secs(2)).await.unwrap();
    assert!(after.url.ends_with("/echo/other.html"), "{}", after.url);
    assert!(after.tools.is_empty());
    env.stop().await;
}

#[tokio::test]
async fn throwing_tool_is_a_tool_failure() {
    let env = Env::start().await;
    let tab = env.open("echo/", false).await;
    tab.wait_for_tools(WAIT).await.unwrap();

    let result = tab.execute("fail", &json!({}), CALL).await;
    assert!(
        matches!(&result, Err(BrowserError::ToolFailed { tool, .. }) if tool == "fail"),
        "{result:?}"
    );
    let missing = tab.execute("nope", &json!({}), CALL).await;
    assert!(
        matches!(&missing, Err(BrowserError::ToolNotFound(name)) if name == "nope"),
        "{missing:?}"
    );
    env.stop().await;
}

#[tokio::test]
async fn legacy_site_needs_the_shim() {
    let env = Env::start().await;

    let plain = env.open("legacy/", false).await;
    let discovery = plain.wait_for_tools(Duration::from_secs(2)).await.unwrap();
    assert!(discovery.tools.is_empty());

    let shimmed = env.open("legacy/", true).await;
    let discovery = shimmed.wait_for_tools(WAIT).await.unwrap();
    let names: Vec<_> = discovery.tools.iter().map(|t| t.name.as_str()).collect();
    // `temporary` is registered and then unregistered by the page.
    assert_eq!(names, ["list_items"]);
    let list = discovery.tool("list_items").unwrap();
    assert_eq!(
        list.input_schema,
        json!({ "type": "object", "properties": {} })
    );
    env.stop().await;
}

#[tokio::test]
async fn alpha_search_products_returns_filtered_items() {
    let env = Env::start().await;
    let tab = env.open("alpha/", false).await;
    let discovery = tab.wait_for_tools(WAIT).await.unwrap();
    let search = discovery.tool("search_products").unwrap();
    assert_eq!(search.input_schema["required"], json!(["query"]));

    let args = json!({ "query": "boots", "category": "SOCCER", "max_price": 150 });
    let outcome = tab.execute("search_products", &args, CALL).await.unwrap();
    let body = parse(outcome.output);
    assert_eq!(body["count"], 1);
    assert_eq!(body["items"][0]["sku"], "A-100");
    env.stop().await;
}
