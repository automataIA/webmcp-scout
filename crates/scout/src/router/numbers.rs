//! Numbers in a request, parsed in code because Jev is weak at numeric
//! precision: `180`, `180€`, `€ 180,00`, `1.299,90`, `under 180`,
//! `EU 45`, `#1234`, ranges like `50-100`. Parts of identifiers (`A-100`,
//! `mp3`) and dates (`2026-11-03`, `3/11`) are not numbers. Numbers written as
//! words ("a hundred") are left to the LLM.

use serde::Serialize;

use crate::schema::ParamKind;

const CONTEXT_WORDS: usize = 3;
const CURRENCY: &[char] = &['€', '$', '£'];
const LEADING: &[char] = &['(', '[', '{', '"', '\'', '#', '~'];
const TRAILING: &[char] = &[',', '.', ';', ':', '!', '?', ')', ']', '}', '"', '\'', '%'];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NumberCandidate {
    pub value: f64,
    /// The token the number came from.
    pub text: String,
    /// A few words around the token, to tell candidates apart.
    pub context: String,
}

impl NumberCandidate {
    /// Whether the value satisfies a numeric parameter's type and bounds.
    pub fn fits(&self, kind: &ParamKind) -> bool {
        let ParamKind::Number {
            integer,
            minimum,
            maximum,
            exclusive_minimum,
            exclusive_maximum,
        } = kind
        else {
            return false;
        };
        let v = self.value;
        !(*integer && v.fract() != 0.0)
            && minimum.is_none_or(|m| v >= m)
            && maximum.is_none_or(|m| v <= m)
            && exclusive_minimum.is_none_or(|m| v > m)
            && exclusive_maximum.is_none_or(|m| v < m)
    }
}

/// Every number in `text`, in order of appearance.
pub fn candidates(text: &str) -> Vec<NumberCandidate> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    for (i, word) in words.iter().enumerate() {
        for value in parse_token(word) {
            let from = i.saturating_sub(CONTEXT_WORDS);
            let to = (i + CONTEXT_WORDS + 1).min(words.len());
            out.push(NumberCandidate {
                value,
                text: (*word).to_owned(),
                context: words[from..to].join(" "),
            });
        }
    }
    out
}

/// A number, a range (two numbers), or nothing.
fn parse_token(word: &str) -> Vec<f64> {
    let token = word
        .trim_start_matches(LEADING)
        .trim_end_matches(TRAILING)
        .trim_matches(CURRENCY)
        .replace('–', "-");
    if token.is_empty() || token.chars().any(|c| c.is_alphabetic() || c == '/') {
        return Vec::new();
    }
    let parts: Vec<&str> = token.split('-').collect();
    match parts.as_slice() {
        [one] => parse_number(one).into_iter().collect(),
        [low, high] => match (parse_number(low), parse_number(high)) {
            (Some(low), Some(high)) if low < high => vec![low, high],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Digits with `.`/`,` separators. The last separator is decimal unless it
/// is followed by exactly three digits and is the only kind used; earlier
/// ones are thousands separators and must group three digits.
fn parse_number(s: &str) -> Option<f64> {
    if !s.starts_with(|c: char| c.is_ascii_digit())
        || !s
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
    {
        return None;
    }
    let groups: Vec<&str> = s.split(['.', ',']).collect();
    if groups.iter().any(|g| g.is_empty()) {
        return None;
    }
    let separators: Vec<char> = s.chars().filter(|c| !c.is_ascii_digit()).collect();
    let Some(&last) = separators.last() else {
        return s.parse().ok();
    };
    let mixed = separators.iter().any(|&c| c != last);
    let tail = groups[groups.len() - 1];
    let decimal = mixed || tail.len() != 3;
    let thousands = if decimal {
        &groups[1..groups.len() - 1]
    } else {
        &groups[1..]
    };
    if thousands.iter().any(|g| g.len() != 3)
        || (decimal && separators[..separators.len() - 1].contains(&last))
    {
        return None;
    }
    let integer: String = if decimal {
        groups[..groups.len() - 1].concat()
    } else {
        groups.concat()
    };
    let number = if decimal {
        format!("{integer}.{tail}")
    } else {
        integer
    };
    number.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(text: &str) -> Vec<f64> {
        candidates(text).iter().map(|c| c.value).collect()
    }

    #[test]
    fn plain_and_currency_amounts() {
        assert_eq!(values("soccer boots under 180"), [180.0]);
        assert_eq!(values("under 180€"), [180.0]);
        assert_eq!(values("max € 180,00"), [180.0]);
        assert_eq!(values("jerseys at most 49,90 €"), [49.9]);
        assert_eq!(values("budget is 30 EUR"), [30.0]);
        assert_eq!(values("up to $1,299.99"), [1299.99]);
        assert_eq!(values("up to 1.299,90 euro"), [1299.9]);
        assert_eq!(values("a 2.5 kg ball"), [2.5]);
        assert_eq!(values("1.000 euro"), [1000.0]);
    }

    #[test]
    fn sizes_ids_and_ranges() {
        assert_eq!(values("shoes EU 45"), [45.0]);
        assert_eq!(
            values("what features does the product with id 42 have?"),
            [42.0]
        );
        assert_eq!(values("cancel my order #1234"), [1234.0]);
        assert_eq!(values("between 50-100 euros"), [50.0, 100.0]);
        assert_eq!(values("from 50 to 100"), [50.0, 100.0]);
    }

    #[test]
    fn identifiers_and_dates_are_not_numbers() {
        assert!(values("show me product A-100").is_empty());
        assert!(values("flights on 2026-11-03").is_empty());
        assert!(values("on 3/11 with an mp3 player").is_empty());
        assert!(values("a hundred euros").is_empty());
        assert!(values("1,2,3 go").is_empty());
        assert!(values("1.2.3").is_empty());
    }

    #[test]
    fn context_and_bounds() {
        let c = &candidates("I want size 42 boots under 80 euros please")[1];
        assert_eq!(c.text, "80");
        assert_eq!(c.context, "42 boots under 80 euros please");

        let price = ParamKind::Number {
            integer: false,
            minimum: Some(0.0),
            maximum: None,
            exclusive_minimum: None,
            exclusive_maximum: Some(1000.0),
        };
        let id = ParamKind::Number {
            integer: true,
            minimum: None,
            maximum: None,
            exclusive_minimum: None,
            exclusive_maximum: None,
        };
        let fits = |text: &str, kind: &ParamKind| candidates(text)[0].fits(kind);
        assert!(fits("49,90", &price) && !fits("49,90", &id));
        assert!(!fits("1.500", &price) && fits("1.500", &id));
    }
}
