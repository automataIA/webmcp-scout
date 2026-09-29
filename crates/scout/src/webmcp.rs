//! WebMCP tool model as seen by the runtime.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A tool registered by the page, with its input schema parsed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredTool {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub input_schema: Value,
    pub origin: String,
}

/// A tool the page registered but the runtime will not use, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectedTool {
    pub name: String,
    pub reason: String,
}

/// Result of listing the tools of the current document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Discovery {
    pub url: String,
    pub origin: String,
    pub tools: Vec<DiscoveredTool>,
    pub rejected: Vec<RejectedTool>,
}

impl Discovery {
    pub fn tool(&self, name: &str) -> Option<&DiscoveredTool> {
        self.tools.iter().find(|t| t.name == name)
    }
}

/// Parses the `inputSchema` exposed by the page. Tools registered without a
/// schema accept no arguments, which the spec models as an empty object schema.
pub(crate) fn parse_input_schema(raw: Option<&str>) -> Result<Value, String> {
    let raw = match raw.map(str::trim) {
        None | Some("") => return Ok(serde_json::json!({ "type": "object", "properties": {} })),
        Some(raw) => raw,
    };
    let schema: Value =
        serde_json::from_str(raw).map_err(|e| format!("inputSchema is not valid JSON: {e}"))?;
    if !schema.is_object() {
        return Err("inputSchema is not a JSON object".into());
    }
    Ok(schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_or_empty_schema_means_no_arguments() {
        let empty = json!({ "type": "object", "properties": {} });
        assert_eq!(parse_input_schema(None).unwrap(), empty);
        assert_eq!(parse_input_schema(Some("  ")).unwrap(), empty);
    }

    #[test]
    fn schema_string_is_parsed() {
        let schema = parse_input_schema(Some(r#"{"type":"object","required":["q"]}"#)).unwrap();
        assert_eq!(schema["required"], json!(["q"]));
    }

    #[test]
    fn invalid_schemas_are_rejected() {
        assert!(parse_input_schema(Some("{not json")).is_err());
        assert!(parse_input_schema(Some("[1,2]")).is_err());
    }
}
