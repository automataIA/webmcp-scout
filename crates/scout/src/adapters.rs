//! Explicit, versioned output adapters. WebMCP publishes no output
//! schema, so each supported tool gets a hand-written parser; any other
//! output is kept as JSON or as a status text, never guessed at.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// Adapter names accepted in `sites.toml`.
pub const KNOWN: &[&str] = &[ALPHA_SEARCH, BETA_FIND, SHOES_ALL, SPORT_SEARCH];

const ALPHA_SEARCH: &str = "alpha.search_products.v1";
const BETA_FIND: &str = "beta.find_shoes.v1";
const SHOES_ALL: &str = "shoes.get_all_store_items.v1";
const SPORT_SEARCH: &str = "sport.search_product.v1";

#[derive(Debug, thiserror::Error)]
#[error("adapter `{adapter}`: {message}")]
pub struct AdapterError {
    pub adapter: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Money {
    pub amount: f64,
    /// ISO 4217 code.
    pub currency: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Product {
    /// Site name from `sites.toml`.
    pub source: String,
    pub id: String,
    pub name: String,
    pub price: Option<Money>,
    /// Units in stock, when the site reports a count (0 = out of stock). A site
    /// that only says "in stock" gives `None`.
    pub availability: Option<u64>,
    pub url: Option<String>,
    /// Site-specific fields kept for filtering and display.
    pub attributes: Map<String, Value>,
}

/// A tool result after adaptation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Output {
    Products(Vec<Product>),
    /// JSON from a tool without an adapter.
    Json(Value),
    /// Non-JSON text, e.g. "No product with SKU X".
    Status(String),
    /// The tool returned nothing.
    Empty,
}

impl Output {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Products(_) => "products",
            Self::Json(_) => "json",
            Self::Status(_) => "status",
            Self::Empty => "empty",
        }
    }
}

/// The site a tool output comes from, as configured in `sites.toml`.
#[derive(Debug, Clone, Copy)]
pub struct Source<'a> {
    pub name: &'a str,
    /// ISO 4217 code of the site's prices.
    pub currency: &'a str,
    /// Page URL, the base for relative product links.
    pub url: &'a str,
}

/// Adapts a raw tool output with the named adapter (one of [`KNOWN`]).
pub fn adapt(
    adapter: Option<&str>,
    source: &Source,
    raw: Option<&str>,
) -> Result<Output, AdapterError> {
    let Some(raw) = raw else {
        return Ok(Output::Empty);
    };
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return Ok(Output::Status(raw.to_owned()));
    };
    let Some(adapter) = adapter else {
        return Ok(Output::Json(value));
    };
    let products = match adapter {
        ALPHA_SEARCH => alpha_search(source, value),
        BETA_FIND => beta_find(source, value),
        SHOES_ALL => shoes_all(source, value),
        SPORT_SEARCH => sport_search(source, value),
        other => {
            return Err(AdapterError {
                adapter: other.to_owned(),
                message: "unknown adapter".into(),
            });
        }
    };
    products.map(Output::Products).map_err(|e| AdapterError {
        adapter: adapter.to_owned(),
        message: format!("unexpected output shape: {e}"),
    })
}

impl Source<'_> {
    fn price(&self, amount: f64) -> Option<Money> {
        Some(Money {
            amount,
            currency: self.currency.to_owned(),
        })
    }

    /// `link` resolved against the page URL; `None` if it does not parse.
    fn link(&self, link: &str) -> Option<String> {
        url::Url::parse(self.url)
            .and_then(|base| base.join(link))
            .ok()
            .map(String::from)
    }
}

/// Fixture `alpha`: `{"count": n, "items": [{sku, title, category, price_eur, sizes, stock}]}`.
fn alpha_search(site: &Source, value: Value) -> Result<Vec<Product>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Response {
        items: Vec<Item>,
    }
    #[derive(Deserialize)]
    struct Item {
        sku: String,
        title: String,
        category: String,
        price_eur: f64,
        sizes: Vec<Value>,
        stock: u64,
    }
    let response: Response = serde_json::from_value(value)?;
    Ok(response
        .items
        .into_iter()
        .map(|i| Product {
            source: site.name.to_owned(),
            id: i.sku,
            name: i.title,
            price: site.price(i.price_eur),
            availability: Some(i.stock),
            url: None,
            attributes: object(json!({ "category": i.category, "sizes": i.sizes })),
        })
        .collect())
}

/// Chrome Labs sport-shop `search_product`:
/// `{"success": true, "message": ..., "count": n, "products": [{id, name, price,
/// category, description, image, size, tags, availability}]}`.
fn sport_search(site: &Source, value: Value) -> Result<Vec<Product>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Response {
        products: Vec<Item>,
    }
    #[derive(Deserialize)]
    struct Item {
        id: String,
        name: String,
        price: f64,
        category: String,
        #[serde(default)]
        size: Option<String>,
        #[serde(default)]
        tags: Vec<String>,
        availability: u64,
    }
    let response: Response = serde_json::from_value(value)?;
    Ok(response
        .products
        .into_iter()
        .map(|i| Product {
            source: site.name.to_owned(),
            id: i.id,
            name: i.name,
            price: site.price(i.price),
            availability: Some(i.availability),
            url: None,
            attributes: object(json!({ "category": i.category, "size": i.size, "tags": i.tags })),
        })
        .collect())
}

/// Fixture `beta`: `{"results": [{code, label, price_cents, sizes_eu, available, link}]}`.
/// Stock is a flag with no count.
fn beta_find(site: &Source, value: Value) -> Result<Vec<Product>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Response {
        results: Vec<Item>,
    }
    #[derive(Deserialize)]
    struct Item {
        code: String,
        label: String,
        price_cents: u64,
        sizes_eu: Vec<String>,
        available: bool,
        link: String,
    }
    let response: Response = serde_json::from_value(value)?;
    Ok(response
        .results
        .into_iter()
        .map(|i| Product {
            source: site.name.to_owned(),
            id: i.code,
            name: i.label,
            price: site.price(i.price_cents as f64 / 100.0),
            availability: (!i.available).then_some(0),
            url: site.link(&i.link),
            attributes: object(json!({ "sizes": i.sizes_eu })),
        })
        .collect())
}

/// Shoe Store `get_all_store_items`: `[{id, name, brand, gender, category,
/// price, is_in_inventory, items_left, imageURL, slug, featured}]`. The
/// whole catalog: filtering is left to the ranking.
fn shoes_all(site: &Source, value: Value) -> Result<Vec<Product>, serde_json::Error> {
    #[derive(Deserialize)]
    struct Item {
        id: u64,
        name: String,
        brand: String,
        gender: String,
        category: String,
        price: f64,
        is_in_inventory: bool,
        items_left: u64,
    }
    let items: Vec<Item> = serde_json::from_value(value)?;
    Ok(items
        .into_iter()
        .map(|i| Product {
            source: site.name.to_owned(),
            id: i.id.to_string(),
            name: i.name,
            price: site.price(i.price),
            availability: Some(if i.is_in_inventory { i.items_left } else { 0 }),
            url: None,
            attributes: object(
                json!({ "brand": i.brand, "gender": i.gender, "category": i.category }),
            ),
        })
        .collect())
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: Source = Source {
        name: "alpha",
        currency: "EUR",
        url: "http://127.0.0.1:8080/alpha/",
    };
    const BETA: Source = Source {
        name: "beta",
        currency: "EUR",
        url: "http://127.0.0.1:8080/beta/",
    };
    const SHOES: Source = Source {
        name: "shoes",
        currency: "USD",
        url: "https://andreinwald.github.io/webmcp-demo/",
    };
    const SPORT: Source = Source {
        name: "sport",
        currency: "USD",
        url: "https://googlechromelabs.github.io/webmcp-tools/demos/sport-shop-angular/",
    };

    #[test]
    fn alpha_search_products() {
        let raw = r#"{"count":1,"items":[{"sku":"A-102","title":"Academy Turf Soccer Shoes","category":"SOCCER","price_eur":79.5,"sizes":[38,39],"stock":0}]}"#;
        let Output::Products(products) = adapt(Some(ALPHA_SEARCH), &ALPHA, Some(raw)).unwrap()
        else {
            panic!("expected products");
        };
        assert_eq!(products.len(), 1);
        let p = &products[0];
        assert_eq!(
            (p.id.as_str(), p.name.as_str()),
            ("A-102", "Academy Turf Soccer Shoes")
        );
        assert_eq!(p.price.as_ref().unwrap().amount, 79.5);
        assert_eq!(p.price.as_ref().unwrap().currency, "EUR");
        assert_eq!(p.availability, Some(0));
        assert_eq!(p.attributes["category"], "SOCCER");
    }

    #[test]
    fn sport_search_products() {
        let raw = r#"{"success":true,"message":"Navigating to search results for \"ball\"","count":1,"products":[{"id":"google-mls-pro-ball","name":"Google Pro Match Ball","price":165,"category":"SOCCER","description":"Official match ball.","image":"assets/products/ball.png","size":"adult","tags":["ball","match"],"availability":2}]}"#;
        let Output::Products(products) = adapt(Some(SPORT_SEARCH), &SPORT, Some(raw)).unwrap()
        else {
            panic!("expected products");
        };
        let p = &products[0];
        assert_eq!(p.id, "google-mls-pro-ball");
        assert_eq!(p.price.as_ref().unwrap().amount, 165.0);
        assert_eq!(p.availability, Some(2));
        assert_eq!(p.attributes["tags"], json!(["ball", "match"]));
    }

    #[test]
    fn beta_find_shoes() {
        let raw = r#"{"results":[{"code":"B-3","label":"Indoor Futsal Soccer Shoes","price_cents":6500,"sizes_eu":["42","44"],"available":false,"link":"p/B-3"},{"code":"B-4","label":"Street Soccer Trainers","price_cents":8900,"sizes_eu":["42"],"available":true,"link":"p/B-4"}]}"#;
        let Output::Products(products) = adapt(Some(BETA_FIND), &BETA, Some(raw)).unwrap() else {
            panic!("expected products");
        };
        assert_eq!(products[0].price.as_ref().unwrap().amount, 65.0);
        assert_eq!(products[0].availability, Some(0));
        assert_eq!(products[1].availability, None);
        assert_eq!(
            products[1].url.as_deref(),
            Some("http://127.0.0.1:8080/beta/p/B-4")
        );
        assert_eq!(products[1].attributes["sizes"], json!(["42"]));
    }

    #[test]
    fn shoes_all_store_items() {
        // Trimmed from a live `get_all_store_items` call.
        let raw = r#"[{"id":1,"name":"Nike React Infinity Run Flyknit","brand":"NIKE","gender":"MEN","category":"RUNNING","price":160,"is_in_inventory":true,"items_left":3,"imageURL":"https://static.nike.com/x.jpg","slug":"nike-react-infinity-run-flyknit","featured":0},{"id":12,"name":"Nike Joyride Dual Run","brand":"NIKE","gender":"KIDS","category":"RUNNING","price":110,"is_in_inventory":false,"items_left":3,"imageURL":"https://static.nike.com/y.jpg","slug":"nike-joyride-dual-run","featured":0}]"#;
        let Output::Products(products) = adapt(Some(SHOES_ALL), &SHOES, Some(raw)).unwrap() else {
            panic!("expected products");
        };
        assert_eq!(products[0].id, "1");
        assert_eq!(products[0].price.as_ref().unwrap().currency, "USD");
        assert_eq!(products[0].availability, Some(3));
        // Not in inventory wins over a stale `items_left`.
        assert_eq!(products[1].availability, Some(0));
        assert_eq!(products[1].attributes["gender"], "KIDS");
    }

    #[test]
    fn text_json_and_empty_outputs_pass_through() {
        assert_eq!(
            adapt(Some(ALPHA_SEARCH), &ALPHA, Some("No product with SKU X")).unwrap(),
            Output::Status("No product with SKU X".into())
        );
        assert_eq!(
            adapt(None, &ALPHA, Some(r#"{"a":1}"#)).unwrap(),
            Output::Json(json!({ "a": 1 }))
        );
        assert_eq!(adapt(None, &ALPHA, None).unwrap(), Output::Empty);
    }

    #[test]
    fn wrong_shape_is_an_error_not_an_empty_list() {
        let err = adapt(Some(SPORT_SEARCH), &SPORT, Some(r#"{"items":[]}"#)).unwrap_err();
        assert_eq!(err.adapter, SPORT_SEARCH);
    }
}
