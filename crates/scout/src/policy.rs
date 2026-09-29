//! Capability allowlist. Only `(origin, tool, schema_hash)` triples
//! written by hand in `sites.toml` may run; everything else is blocked. Tool
//! metadata, schemas and outputs never change this configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::adapters;

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("{path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("invalid sites file: {0}")]
    Parse(String),
    #[error("site `{site}`: {message}")]
    Invalid { site: String, message: String },
    #[error("unknown site `{0}`")]
    UnknownSite(String),
}

/// Why a tool may not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PolicyViolation {
    #[error("page origin {actual} is not the site origin {expected}")]
    OriginMismatch { expected: String, actual: String },
    #[error("tool `{0}` is not in the allowlist")]
    NotAllowlisted(String),
    #[error("tool `{tool}` changed its input schema: allowlisted {expected}, found {actual}")]
    SchemaChanged {
        tool: String,
        expected: String,
        actual: String,
    },
}

/// Side-effect class of a capability. The MVP only allows read-only tools,
/// so any other class in `sites.toml` fails to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityClass {
    ReadOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    pub tool: String,
    /// `sha256:<hex>` as printed by `scout tools`.
    pub schema_hash: String,
    pub class: CapabilityClass,
    /// Output adapter (see [`adapters::KNOWN`]); without one the raw output
    /// is kept as JSON or text.
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SitesFile {
    #[serde(default)]
    sites: BTreeMap<String, SiteConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteConfig {
    pub url: String,
    #[serde(default)]
    pub legacy_shim: bool,
    /// ISO 4217 code for prices the site reports without a currency.
    pub currency: String,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
}

/// A validated site entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Site {
    pub name: String,
    pub url: String,
    /// Serialized origin of `url`, e.g. `https://example.com`.
    pub origin: String,
    pub legacy_shim: bool,
    pub currency: String,
    pub capabilities: Vec<Capability>,
}

impl Site {
    pub fn new(name: &str, config: SiteConfig) -> Result<Self, PolicyError> {
        let invalid = |message: String| PolicyError::Invalid {
            site: name.to_owned(),
            message,
        };
        let url = url::Url::parse(&config.url).map_err(|e| invalid(format!("url: {e}")))?;
        let origin = url.origin();
        if !origin.is_tuple() {
            return Err(invalid(format!("url {url} has no origin")));
        }
        let mut tools = BTreeSet::new();
        for cap in &config.capabilities {
            if !tools.insert(cap.tool.as_str()) {
                return Err(invalid(format!("tool `{}` listed twice", cap.tool)));
            }
            let hex = cap.schema_hash.strip_prefix("sha256:").unwrap_or("");
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(invalid(format!(
                    "tool `{}`: schema_hash must be `sha256:<64 hex digits>`",
                    cap.tool
                )));
            }
            if let Some(adapter) = &cap.adapter
                && !adapters::KNOWN.contains(&adapter.as_str())
            {
                return Err(invalid(format!(
                    "tool `{}`: unknown adapter `{adapter}`",
                    cap.tool
                )));
            }
        }
        Ok(Self {
            name: name.to_owned(),
            url: config.url,
            origin: origin.ascii_serialization(),
            legacy_shim: config.legacy_shim,
            currency: config.currency,
            capabilities: config.capabilities,
        })
    }

    /// The capability that authorizes `tool` on a page of `origin` whose
    /// current input schema hashes to `schema_hash`.
    pub fn check(
        &self,
        origin: &str,
        tool: &str,
        schema_hash: &str,
    ) -> Result<&Capability, PolicyViolation> {
        if origin != self.origin {
            return Err(PolicyViolation::OriginMismatch {
                expected: self.origin.clone(),
                actual: origin.to_owned(),
            });
        }
        let cap = self
            .capabilities
            .iter()
            .find(|c| c.tool == tool)
            .ok_or_else(|| PolicyViolation::NotAllowlisted(tool.to_owned()))?;
        if cap.schema_hash != schema_hash {
            return Err(PolicyViolation::SchemaChanged {
                tool: tool.to_owned(),
                expected: cap.schema_hash.clone(),
                actual: schema_hash.to_owned(),
            });
        }
        Ok(cap)
    }
}

/// Sites from a `sites.toml`, by name.
#[derive(Debug, Clone, Default)]
pub struct Sites(BTreeMap<String, Site>);

impl Sites {
    pub fn load(path: &Path) -> Result<Self, PolicyError> {
        let text = std::fs::read_to_string(path).map_err(|e| PolicyError::Io {
            path: path.to_owned(),
            message: e.to_string(),
        })?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, PolicyError> {
        let file: SitesFile =
            toml::from_str(text).map_err(|e| PolicyError::Parse(e.to_string()))?;
        file.sites
            .into_iter()
            .map(|(name, config)| Ok((name.clone(), Site::new(&name, config)?)))
            .collect::<Result<_, _>>()
            .map(Self)
    }

    pub fn get(&self, name: &str) -> Result<&Site, PolicyError> {
        self.0
            .get(name)
            .ok_or_else(|| PolicyError::UnknownSite(name.to_owned()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000abc";

    fn sites(extra: &str) -> Result<Sites, PolicyError> {
        Sites::parse(&format!(
            r#"
[sites.shop]
url = "https://shop.example/demo/index.html"
currency = "EUR"

[[sites.shop.capabilities]]
tool = "search"
schema_hash = "{HASH}"
class = "read_only"
adapter = "alpha.search_products.v1"
{extra}"#
        ))
    }

    #[test]
    fn allowlisted_triple_passes() {
        let sites = sites("").unwrap();
        let shop = sites.get("shop").unwrap();
        assert_eq!(shop.origin, "https://shop.example");
        let cap = shop.check("https://shop.example", "search", HASH).unwrap();
        assert_eq!(cap.adapter.as_deref(), Some("alpha.search_products.v1"));
        assert!(matches!(
            sites.get("other"),
            Err(PolicyError::UnknownSite(_))
        ));
    }

    #[test]
    fn origin_tool_and_hash_are_all_checked() {
        let sites = sites("").unwrap();
        let shop = sites.get("shop").unwrap();
        assert!(matches!(
            shop.check("https://evil.example", "search", HASH),
            Err(PolicyViolation::OriginMismatch { .. })
        ));
        assert_eq!(
            shop.check("https://shop.example", "buy", HASH),
            Err(PolicyViolation::NotAllowlisted("buy".into()))
        );
        assert!(matches!(
            shop.check("https://shop.example", "search", "sha256:beef"),
            Err(PolicyViolation::SchemaChanged { .. })
        ));
    }

    #[test]
    fn invalid_entries_fail_to_load() {
        let duplicate = format!(
            "[[sites.shop.capabilities]]\ntool = \"search\"\nschema_hash = \"{HASH}\"\nclass = \"read_only\"\n"
        );
        let write_class = "[[sites.shop.capabilities]]\ntool = \"buy\"\nschema_hash = \"sha256:1\"\nclass = \"write\"\n";
        let short_hash = "[[sites.shop.capabilities]]\ntool = \"buy\"\nschema_hash = \"sha256:1\"\nclass = \"read_only\"\n";
        let bad_adapter = format!(
            "[[sites.shop.capabilities]]\ntool = \"get\"\nschema_hash = \"{HASH}\"\nclass = \"read_only\"\nadapter = \"nope\"\n"
        );
        for extra in [
            duplicate.as_str(),
            write_class,
            short_hash,
            bad_adapter.as_str(),
        ] {
            assert!(sites(extra).is_err(), "accepted: {extra}");
        }
        assert!(Sites::parse("[sites.x]\nurl = \"not a url\"\ncurrency = \"EUR\"\n").is_err());
    }
}
