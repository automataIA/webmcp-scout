//! Ranking across sites: hard filters in code (stock, price, size),
//! then a deterministic score. The constraints are read from the user request
//! only, never from page content or tool output.

use std::cmp::Ordering;

use serde::Serialize;
use serde_json::Value;

use crate::adapters::Product;
use crate::router::numbers;

/// Words right before a number that make it a price ceiling.
const PRICE_CUES: &[&str] = &["under", "below", "max", "maximum", "within", "budget"];
const PRICE_CUES_2: &[&str] = &["less than", "up to", "at most", "no more than"];
/// Words right before a number that make it a shoe or clothing size.
const SIZE_CUES: &[&str] = &["size", "eu"];
const STOPWORDS: &[&str] = &[
    "the", "all", "and", "for", "with", "that", "some", "any", "find", "show", "search", "want",
    "need", "looking", "under", "below", "less", "than", "max", "maximum", "most", "within",
    "budget", "size", "price", "euro", "eur", "dollar", "usd", "pair", "please",
];

/// What the user asked for, as far as code can tell.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Constraints {
    pub max_price: Option<f64>,
    /// ISO 4217 code of `max_price`, when the request names one.
    pub currency: Option<String>,
    pub size: Option<String>,
}

impl Constraints {
    pub fn from_request(text: &str) -> Self {
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
        let mut out = Self::default();
        for (i, word) in words.iter().enumerate() {
            let Some(value) = numbers::candidates(word).first().map(|c| c.value) else {
                continue;
            };
            let prev1 = i.checked_sub(1).map(|j| bare(&words[j]));
            let prev2 = i
                .checked_sub(2)
                .map(|j| format!("{} {}", bare(&words[j]), bare(&words[j + 1])));
            if prev1.is_some_and(|w| SIZE_CUES.contains(&w)) {
                out.size.get_or_insert_with(|| format_number(value));
            } else if prev1.is_some_and(|w| PRICE_CUES.contains(&w))
                || prev2.is_some_and(|w| PRICE_CUES_2.contains(&w.as_str()))
            {
                out.max_price = Some(out.max_price.map_or(value, |m| m.min(value)));
                let next = words.get(i + 1).map(|w| bare(w));
                if let Some(code) = currency(word).or_else(|| next.and_then(currency)) {
                    out.currency = Some(code.to_owned());
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ranked {
    /// Share of the keywords found in the product text, 0 to 1.
    pub score: f64,
    pub product: Product,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Excluded {
    pub source: String,
    pub id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ranking {
    pub constraints: Constraints,
    pub keywords: Vec<String>,
    /// Best first.
    pub products: Vec<Ranked>,
    pub excluded: Vec<Excluded>,
}

/// Keywords for scoring: lowercase words of `texts`, without stopwords,
/// numbers and a plural `s`, deduplicated in order.
pub fn keywords<'a>(texts: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for text in texts {
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if word.chars().count() < 3
                || word.chars().all(|c| c.is_ascii_digit())
                || STOPWORDS.contains(&word)
            {
                continue;
            }
            let word = match word.strip_suffix('s') {
                Some(stem) if stem.chars().count() >= 3 => stem,
                _ => word,
            };
            if !STOPWORDS.contains(&word) && !out.iter().any(|k| k == word) {
                out.push(word.to_owned());
            }
        }
    }
    out
}

/// Drops the products that break a constraint, then orders the rest by
/// score, price (unknown last), source and id.
pub fn rank(products: Vec<Product>, constraints: &Constraints, keywords: &[String]) -> Ranking {
    let mut kept = Vec::new();
    let mut excluded = Vec::new();
    for product in products {
        match violation(&product, constraints) {
            Some(reason) => excluded.push(Excluded {
                source: product.source,
                id: product.id,
                reason,
            }),
            None => kept.push(Ranked {
                score: score(&product, keywords),
                product,
            }),
        }
    }
    kept.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| price_order(&a.product, &b.product))
            .then_with(|| a.product.source.cmp(&b.product.source))
            .then_with(|| a.product.id.cmp(&b.product.id))
    });
    Ranking {
        constraints: constraints.clone(),
        keywords: keywords.to_vec(),
        products: kept,
        excluded,
    }
}

fn violation(product: &Product, c: &Constraints) -> Option<String> {
    if product.availability == Some(0) {
        return Some("out of stock".into());
    }
    if let Some(max) = c.max_price {
        let Some(price) = &product.price else {
            return Some("no price".into());
        };
        if let Some(currency) = c.currency.as_deref().filter(|&c| c != price.currency) {
            return Some(format!(
                "price in {}, budget in {currency} (no conversion)",
                price.currency
            ));
        }
        if price.amount > max {
            return Some(format!(
                "price {:.2} {} over the budget of {}",
                price.amount,
                price.currency,
                format_number(max)
            ));
        }
    }
    if let Some(size) = &c.size
        && let Some(Value::Array(sizes)) = product.attributes.get("sizes")
        && !sizes.iter().any(|s| size_text(s).as_deref() == Some(size))
    {
        return Some(format!("size {size} not listed"));
    }
    None
}

fn score(product: &Product, keywords: &[String]) -> f64 {
    if keywords.is_empty() {
        return 0.0;
    }
    let mut text = product.name.to_lowercase();
    for value in product.attributes.values() {
        match value {
            Value::String(s) => push_word(&mut text, s),
            Value::Array(items) => items
                .iter()
                .filter_map(Value::as_str)
                .for_each(|s| push_word(&mut text, s)),
            _ => {}
        }
    }
    // A keyword matches the start of a word: `boot` finds "Boots", `all` does
    // not find "football".
    let words: Vec<&str> = text.split(|c: char| !c.is_alphanumeric()).collect();
    let found = keywords
        .iter()
        .filter(|k| words.iter().any(|w| w.starts_with(k.as_str())))
        .count();
    // Two decimals keep the order stable and the output readable.
    (found as f64 / keywords.len() as f64 * 100.0).round() / 100.0
}

fn push_word(text: &mut String, word: &str) {
    text.push(' ');
    text.push_str(&word.to_lowercase());
}

fn price_order(a: &Product, b: &Product) -> Ordering {
    match (&a.price, &b.price) {
        (Some(a), Some(b)) => a.amount.total_cmp(&b.amount),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// A word without surrounding punctuation.
fn bare(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric() && !"€$£".contains(c))
}

fn currency(word: &str) -> Option<&'static str> {
    if word.contains('€') || ["euro", "eur", "euros"].contains(&word) {
        Some("EUR")
    } else if word.contains('$') || ["usd", "dollar", "dollars"].contains(&word) {
        Some("USD")
    } else if word.contains('£') || ["gbp", "pound", "pounds", "sterline"].contains(&word) {
        Some("GBP")
    } else {
        None
    }
}

fn size_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => n.as_f64().map(format_number),
        _ => None,
    }
}

/// `42` rather than `42.0`.
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::adapters::Money;

    fn product(
        source: &str,
        id: &str,
        name: &str,
        price: Option<f64>,
        stock: Option<u64>,
    ) -> Product {
        Product {
            source: source.into(),
            id: id.into(),
            name: name.into(),
            price: price.map(|amount| Money {
                amount,
                currency: "EUR".into(),
            }),
            availability: stock,
            url: None,
            attributes: serde_json::Map::new(),
        }
    }

    #[test]
    fn constraints_from_requests() {
        let c = Constraints::from_request("soccer boots under 150 euro, size 42");
        assert_eq!(
            c,
            Constraints {
                max_price: Some(150.0),
                currency: Some("EUR".into()),
                size: Some("42".into()),
            }
        );
        let c = Constraints::from_request("football boots size 44 up to 120€");
        assert_eq!(c.max_price, Some(120.0));
        assert_eq!(c.currency.as_deref(), Some("EUR"));
        assert_eq!(c.size.as_deref(), Some("44"));
        let c = Constraints::from_request("running shoes less than $99.50");
        assert_eq!(
            (c.max_price, c.currency.as_deref()),
            (Some(99.5), Some("USD"))
        );
        // A number without a cue is not a constraint.
        assert_eq!(
            Constraints::from_request("show me the 2 best balls"),
            Constraints::default()
        );
    }

    #[test]
    fn keywords_drop_stopwords_numbers_and_plurals() {
        assert_eq!(
            keywords(["find all soccer boots under 150 euros", "soccer boots"]),
            ["soccer", "boot"]
        );
    }

    #[test]
    fn hard_filters_then_score_then_price() {
        let mut small = product("beta", "B-2", "Match Soccer Boots", Some(99.0), None);
        small.attributes.insert("sizes".into(), json!(["39", "40"]));
        let mut fits = product(
            "alpha",
            "A-100",
            "Striker Soccer Boots",
            Some(149.9),
            Some(12),
        );
        fits.attributes.insert("sizes".into(), json!([41, 42]));
        let mut usd = product("sport", "s-1", "Soccer Boots", Some(90.0), Some(4));
        usd.price.as_mut().unwrap().currency = "USD".into();
        let products = vec![
            product("alpha", "A-101", "Pro Soccer Boots", Some(219.0), Some(3)),
            product("alpha", "A-102", "Turf Soccer Shoes", Some(79.5), Some(0)),
            small,
            fits,
            usd,
            product("beta", "B-4", "Street Soccer Trainers", Some(89.0), None),
            product("beta", "B-9", "Soccer Boots", None, None),
        ];
        let c = Constraints::from_request("soccer boots under 150 eur size 42");
        let ranking = rank(products, &c, &keywords(["soccer boots"]));
        let order: Vec<(&str, f64)> = ranking
            .products
            .iter()
            .map(|r| (r.product.id.as_str(), r.score))
            .collect();
        // Sizes unknown (B-4) are kept; a better score beats a lower price.
        assert_eq!(order, [("A-100", 1.0), ("B-4", 0.5)]);
        let reasons: Vec<(&str, &str)> = ranking
            .excluded
            .iter()
            .map(|e| (e.id.as_str(), e.reason.as_str()))
            .collect();
        assert_eq!(
            reasons,
            [
                ("A-101", "price 219.00 EUR over the budget of 150"),
                ("A-102", "out of stock"),
                ("B-2", "size 42 not listed"),
                ("s-1", "price in USD, budget in EUR (no conversion)"),
                ("B-9", "no price"),
            ]
        );
    }

    #[test]
    fn keywords_match_word_starts_in_name_and_attributes() {
        let mut p = product("shoes", "7", "Nike Phantom Venom", Some(80.0), Some(3));
        p.attributes.insert("category".into(), json!("FOOTBALL"));
        p.attributes
            .insert("tags".into(), json!(["boots", "firm-ground"]));
        let k = |words: &[&str]| words.iter().map(|w| w.to_string()).collect::<Vec<_>>();
        assert_eq!(score(&p, &k(&["football", "boot", "ground"])), 1.0);
        assert_eq!(score(&p, &k(&["ball", "oot"])), 0.0);
    }

    #[test]
    fn without_constraints_only_stock_filters() {
        let products = vec![
            product("a", "2", "Ball", Some(20.0), Some(1)),
            product("a", "1", "Ball", Some(10.0), Some(1)),
            product("a", "3", "Ball", None, Some(0)),
        ];
        let ranking = rank(products, &Constraints::default(), &[]);
        let ids: Vec<&str> = ranking
            .products
            .iter()
            .map(|r| r.product.id.as_str())
            .collect();
        assert_eq!(ids, ["1", "2"]);
        assert_eq!(ranking.excluded.len(), 1);
    }
}
