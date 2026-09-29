//! Browser runtime against public WebMCP demos. Needs network access:
//! `cargo test -p scout --test live -- --ignored`. Only read-only tools are
//! executed, with the argument encoding already verified on the echo fixture.

use std::time::Duration;

use scout::browser::{BrowserOptions, ChromeBrowser, TabOptions};
use serde_json::{Value, json};

const SPORT_SHOP: &str =
    "https://googlechromelabs.github.io/webmcp-tools/demos/sport-shop-angular/";
const SHOE_STORE: &str = "https://andreinwald.github.io/webmcp-demo/";
const WAIT: Duration = Duration::from_secs(30);
const CALL: Duration = Duration::from_secs(20);

fn names(discovery: &scout::webmcp::Discovery) -> Vec<&str> {
    discovery.tools.iter().map(|t| t.name.as_str()).collect()
}

#[tokio::test]
#[ignore = "live: needs network"]
async fn sport_shop_search_product() {
    let browser = ChromeBrowser::launch(&BrowserOptions::default())
        .await
        .unwrap();
    let tab = browser
        .open(SPORT_SHOP, &TabOptions::default())
        .await
        .unwrap();
    let discovery = tab.wait_for_tools(WAIT).await.unwrap();
    let names = names(&discovery);
    for expected in ["search_product", "get_product_info", "view_product"] {
        assert!(names.contains(&expected), "{expected} missing: {names:?}");
    }
    let search = discovery.tool("search_product").unwrap();
    assert!(search.input_schema["properties"]["category"]["enum"].is_array());

    let args = json!({ "query": "soccer boots", "category": "SOCCER", "size": "ALL" });
    let outcome = tab.execute("search_product", &args, CALL).await.unwrap();
    let output = outcome.output.expect("search_product output");
    serde_json::from_str::<Value>(&output).expect("search_product returns JSON");
    browser.shutdown().await;
}

#[tokio::test]
#[ignore = "live: needs network"]
async fn shoe_store_with_legacy_shim() {
    let browser = ChromeBrowser::launch(&BrowserOptions::default())
        .await
        .unwrap();
    let tab = browser
        .open(SHOE_STORE, &TabOptions { legacy_shim: true })
        .await
        .unwrap();
    let discovery = tab.wait_for_tools(WAIT).await.unwrap();
    let names = names(&discovery);
    for expected in [
        "get_all_store_items",
        "filter_by_brand",
        "get_filter_options",
    ] {
        assert!(names.contains(&expected), "{expected} missing: {names:?}");
    }

    let outcome = tab
        .execute("get_all_store_items", &json!({}), CALL)
        .await
        .unwrap();
    assert!(outcome.output.is_some_and(|o| !o.is_empty()));
    browser.shutdown().await;
}
