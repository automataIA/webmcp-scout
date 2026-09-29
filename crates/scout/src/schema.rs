//! WebMCP `inputSchema` → [`ToolSpec`]: the explicit JSON Schema subset the
//! router supports. Constructs outside the subset are reported with
//! a reason, never dropped silently.

use serde::Serialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::webmcp::DiscoveredTool;

/// Keywords whose semantics the router cannot honor. Present on a parameter,
/// they make it unsupported; at the top level, the whole tool.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "patternProperties",
];

/// Top-level keywords that carry no constraint the router has to enforce.
const TOP_LEVEL_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "title",
    "description",
];

const MAX_REF_DEPTH: usize = 8;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// `sha256:<hex>` of the canonical (key-sorted) input schema.
    pub schema_hash: String,
    /// The schema as published by the site, used to validate arguments.
    pub input_schema: Value,
    pub params: Vec<ParamSpec>,
    pub support: Support,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParamSpec {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
    /// The schema also accepts `null`.
    pub nullable: bool,
    pub default: Option<Value>,
    pub kind: ParamKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamKind {
    Enum {
        values: Vec<Value>,
    },
    Bool,
    /// Array whose items come from a closed set.
    EnumSet {
        values: Vec<Value>,
        min_items: Option<u64>,
        max_items: Option<u64>,
        unique: bool,
    },
    Number {
        integer: bool,
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: Option<f64>,
        exclusive_maximum: Option<f64>,
    },
    FreeString {
        format: Option<String>,
        pattern: Option<String>,
        max_length: Option<u64>,
    },
    Unsupported {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "level", content = "reasons", rename_all = "snake_case")]
pub enum Support {
    Full,
    /// Usable; the listed optional parts are ignored.
    Partial(Vec<String>),
    /// Not routable: a required part is outside the subset.
    Unsupported(Vec<String>),
}

impl ToolSpec {
    pub fn from_tool(tool: &DiscoveredTool) -> Self {
        let schema = &tool.input_schema;
        let mut blocking = Vec::new();
        let mut ignored = Vec::new();

        match schema.get("type") {
            None => {}
            Some(t) if t == "object" => {}
            Some(t) => blocking.push(format!("top-level type {t} is not an object")),
        }
        if let Some(obj) = schema.as_object() {
            for key in obj.keys() {
                if UNSUPPORTED_KEYWORDS.contains(&key.as_str())
                    || matches!(key.as_str(), "anyOf" | "oneOf")
                {
                    blocking.push(format!("top-level `{key}` is not supported"));
                } else if !TOP_LEVEL_KEYWORDS.contains(&key.as_str()) {
                    ignored.push(format!("top-level `{key}` is ignored"));
                }
            }
        }

        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut params = Vec::new();
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            for (name, prop) in props {
                let param = ParamSpec::new(name, prop, required.contains(&name.as_str()), schema);
                if let ParamKind::Unsupported { reason } = &param.kind {
                    let msg = format!("parameter `{name}`: {reason}");
                    if param.required {
                        blocking.push(msg);
                    } else {
                        ignored.push(msg);
                    }
                }
                params.push(param);
            }
        }
        for name in &required {
            if !params.iter().any(|p| p.name == *name) {
                blocking.push(format!("required parameter `{name}` has no schema"));
            }
        }

        let support = if !blocking.is_empty() {
            blocking.extend(ignored);
            Support::Unsupported(blocking)
        } else if !ignored.is_empty() {
            Support::Partial(ignored)
        } else {
            Support::Full
        };
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            schema_hash: schema_hash(schema),
            input_schema: schema.clone(),
            params,
            support,
        }
    }

    pub fn is_routable(&self) -> bool {
        !matches!(self.support, Support::Unsupported(_))
    }

    /// Argument schema in the form strict structured outputs accept: every
    /// supported parameter listed and required, optional ones nullable,
    /// `additionalProperties: false`. Numeric and string constraints move to
    /// `description` and are enforced by [`ToolSpec::validate_args`].
    pub fn strict_args_schema(&self) -> Value {
        let mut properties = Map::new();
        for param in &self.params {
            let Some(mut schema) = param.kind.strict_schema() else {
                continue;
            };
            if !param.required || param.nullable {
                schema = json!({ "anyOf": [schema, { "type": "null" }] });
            }
            properties.insert(param.name.clone(), schema);
        }
        let required: Vec<&String> = properties.keys().collect();
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    }

    /// Strict schema for the `names` parameters only, all nullable: `null`
    /// means the request does not state a value. Used by the hybrid router's
    /// LLM call for the fields Jev and the parser do not decide.
    pub fn strict_fields_schema(&self, names: &[&str]) -> Value {
        let mut properties = Map::new();
        for param in &self.params {
            if !names.contains(&param.name.as_str()) {
                continue;
            }
            if let Some(schema) = param.kind.strict_schema() {
                properties.insert(
                    param.name.clone(),
                    json!({ "anyOf": [schema, { "type": "null" }] }),
                );
            }
        }
        let required: Vec<&String> = properties.keys().collect();
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    }

    /// Validates arguments against the schema published by the site. External
    /// `$ref`s are never fetched.
    pub fn validate_args(&self, args: &Value) -> Result<(), Vec<String>> {
        let validator = jsonschema::options()
            .build(&self.input_schema)
            .map_err(|e| vec![format!("invalid input schema: {e}")])?;
        let errors: Vec<String> = validator
            .iter_errors(args)
            .map(|e| {
                let path = e.instance_path().to_string();
                if path.is_empty() {
                    e.to_string()
                } else {
                    format!("{path}: {e}")
                }
            })
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl ParamSpec {
    fn new(name: &str, schema: &Value, required: bool, root: &Value) -> Self {
        let (kind, nullable) = match resolve(schema, root).and_then(|s| classify(s, root)) {
            Ok(classified) => classified,
            Err(reason) => (ParamKind::Unsupported { reason }, false),
        };
        Self {
            name: name.to_owned(),
            description: schema
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            required,
            nullable,
            default: schema.get("default").cloned(),
            kind,
        }
    }
}

impl ParamKind {
    fn strict_schema(&self) -> Option<Value> {
        let (mut schema, notes) = match self {
            Self::Enum { values } => (enum_schema(values), Vec::new()),
            Self::Bool => (json!({ "type": "boolean" }), Vec::new()),
            Self::EnumSet {
                values,
                min_items,
                max_items,
                unique,
            } => {
                let mut notes = Vec::new();
                if let Some(n) = min_items {
                    notes.push(format!("at least {n} items"));
                }
                if let Some(n) = max_items {
                    notes.push(format!("at most {n} items"));
                }
                if *unique {
                    notes.push("no duplicates".to_owned());
                }
                (
                    json!({ "type": "array", "items": enum_schema(values) }),
                    notes,
                )
            }
            Self::Number {
                integer,
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
            } => {
                let mut notes = Vec::new();
                if let Some(v) = minimum {
                    notes.push(format!(">= {v}"));
                }
                if let Some(v) = exclusive_minimum {
                    notes.push(format!("> {v}"));
                }
                if let Some(v) = maximum {
                    notes.push(format!("<= {v}"));
                }
                if let Some(v) = exclusive_maximum {
                    notes.push(format!("< {v}"));
                }
                let ty = if *integer { "integer" } else { "number" };
                (json!({ "type": ty }), notes)
            }
            Self::FreeString {
                format,
                pattern,
                max_length,
            } => {
                let mut notes = Vec::new();
                if let Some(f) = format {
                    notes.push(format!("format {f}"));
                }
                if let Some(p) = pattern {
                    notes.push(format!("pattern {p}"));
                }
                if let Some(n) = max_length {
                    notes.push(format!("at most {n} characters"));
                }
                (json!({ "type": "string" }), notes)
            }
            Self::Unsupported { .. } => return None,
        };
        if !notes.is_empty() {
            schema["description"] = Value::String(notes.join(", "));
        }
        Some(schema)
    }
}

fn enum_schema(values: &[Value]) -> Value {
    if values.iter().all(Value::is_string) {
        json!({ "type": "string", "enum": values })
    } else {
        json!({ "enum": values })
    }
}

/// Follows local `$ref`s (`#/...`). External references are unsupported.
fn resolve<'a>(schema: &'a Value, root: &'a Value) -> Result<&'a Value, String> {
    let mut current = schema;
    for _ in 0..MAX_REF_DEPTH {
        let Some(reference) = current.get("$ref").and_then(Value::as_str) else {
            return Ok(current);
        };
        let pointer = reference
            .strip_prefix('#')
            .ok_or_else(|| format!("external $ref `{reference}`"))?;
        current = root
            .pointer(pointer)
            .ok_or_else(|| format!("unresolved $ref `{reference}`"))?;
    }
    Err("$ref chain too deep".to_owned())
}

/// Classifies a parameter schema. Returns the kind and whether `null` is
/// also accepted.
fn classify(schema: &Value, root: &Value) -> Result<(ParamKind, bool), String> {
    if let Some(key) = UNSUPPORTED_KEYWORDS
        .iter()
        .find(|k| schema.get(**k).is_some())
    {
        return Err(format!("`{key}` is not supported"));
    }

    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return enum_kind(values.clone());
    }
    if let Some(value) = schema.get("const") {
        return enum_kind(vec![value.clone()]);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(key).and_then(Value::as_array) {
            return classify_union(key, branches, root);
        }
    }

    let (ty, nullable) = match schema.get("type") {
        Some(Value::String(t)) => (t.as_str(), false),
        Some(Value::Array(types)) => {
            let non_null: Vec<&str> = types
                .iter()
                .filter_map(Value::as_str)
                .filter(|t| *t != "null")
                .collect();
            match non_null.as_slice() {
                [t] => (*t, non_null.len() < types.len()),
                _ => return Err(format!("union type {}", Value::Array(types.clone()))),
            }
        }
        _ => return Err("missing `type`".to_owned()),
    };

    let kind = match ty {
        "boolean" => ParamKind::Bool,
        "number" | "integer" => ParamKind::Number {
            integer: ty == "integer",
            minimum: schema.get("minimum").and_then(Value::as_f64),
            maximum: schema.get("maximum").and_then(Value::as_f64),
            exclusive_minimum: schema.get("exclusiveMinimum").and_then(Value::as_f64),
            exclusive_maximum: schema.get("exclusiveMaximum").and_then(Value::as_f64),
        },
        "string" => ParamKind::FreeString {
            format: schema
                .get("format")
                .and_then(Value::as_str)
                .map(str::to_owned),
            pattern: schema
                .get("pattern")
                .and_then(Value::as_str)
                .map(str::to_owned),
            max_length: schema.get("maxLength").and_then(Value::as_u64),
        },
        "array" => {
            let items = schema
                .get("items")
                .ok_or("array without `items`")
                .and_then(|items| resolve(items, root).map_err(|_| "unresolved `items`"))?;
            match classify(items, root) {
                Ok((ParamKind::Enum { values }, _)) => ParamKind::EnumSet {
                    values,
                    min_items: schema.get("minItems").and_then(Value::as_u64),
                    max_items: schema.get("maxItems").and_then(Value::as_u64),
                    unique: schema
                        .get("uniqueItems")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                },
                _ => return Err("array items are not a closed set".to_owned()),
            }
        }
        "object" => return Err("nested object".to_owned()),
        other => return Err(format!("type `{other}`")),
    };
    Ok((kind, nullable))
}

fn enum_kind(values: Vec<Value>) -> Result<(ParamKind, bool), String> {
    let nullable = values.iter().any(Value::is_null);
    let values: Vec<Value> = values.into_iter().filter(|v| !v.is_null()).collect();
    if values.is_empty() {
        return Err("empty enum".to_owned());
    }
    Ok((ParamKind::Enum { values }, nullable))
}

/// `anyOf`/`oneOf` is supported when every branch is a constant (an enum in
/// disguise), or when it is `[schema, {"type": "null"}]`.
fn classify_union(
    key: &str,
    branches: &[Value],
    root: &Value,
) -> Result<(ParamKind, bool), String> {
    let branches = branches
        .iter()
        .map(|b| resolve(b, root))
        .collect::<Result<Vec<_>, _>>()?;
    let is_null = |b: &&Value| b.get("type").is_some_and(|t| t == "null");
    let nullable = branches.iter().any(is_null);
    let rest: Vec<&Value> = branches.iter().copied().filter(|b| !is_null(b)).collect();

    let consts: Option<Vec<Value>> = rest.iter().map(|b| b.get("const").cloned()).collect();
    if let Some(values) = consts {
        return enum_kind(values).map(|(kind, n)| (kind, n || nullable));
    }
    match rest.as_slice() {
        [single] if nullable => classify(single, root).map(|(kind, _)| (kind, true)),
        _ => Err(format!("`{key}` of non-constant schemas")),
    }
}

/// `sha256:<hex>` of `schema` serialized with object keys sorted, so the hash
/// does not depend on key order.
pub fn schema_hash(schema: &Value) -> String {
    format!("sha256:{}", canonical_sha256(schema))
}

/// Hex SHA-256 of `value` serialized with object keys sorted.
pub(crate) fn canonical_sha256(value: &Value) -> String {
    let digest = Sha256::digest(canonical_json(value).as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let body: Vec<String> = entries
                .into_iter()
                .map(|(k, v)| format!("{}:{}", Value::String(k.clone()), canonical_json(v)))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(schema: Value) -> ToolSpec {
        ToolSpec::from_tool(&DiscoveredTool {
            name: "t".into(),
            title: None,
            description: "test tool".into(),
            input_schema: schema,
            origin: "https://example.test".into(),
        })
    }

    fn kind<'a>(spec: &'a ToolSpec, name: &str) -> &'a ParamKind {
        &spec.params.iter().find(|p| p.name == name).unwrap().kind
    }

    #[test]
    fn sport_shop_search_is_fully_supported() {
        let s = spec(json!({
            "type": "object",
            "properties": {
                "category": { "type": "string", "enum": ["ALL", "SOCCER"] },
                "query": { "type": "string", "description": "Search term" },
                "size": { "type": "string", "enum": ["ALL", "adult", "child"] }
            }
        }));
        assert_eq!(s.support, Support::Full);
        assert_eq!(
            kind(&s, "category"),
            &ParamKind::Enum {
                values: vec![json!("ALL"), json!("SOCCER")]
            }
        );
        assert!(matches!(kind(&s, "query"), ParamKind::FreeString { .. }));
        assert!(s.params.iter().all(|p| !p.required));
    }

    #[test]
    fn numbers_keep_their_bounds() {
        let s = spec(json!({
            "properties": { "max_price": { "type": "number", "minimum": 0 } },
            "required": ["max_price"]
        }));
        assert_eq!(
            kind(&s, "max_price"),
            &ParamKind::Number {
                integer: false,
                minimum: Some(0.0),
                maximum: None,
                exclusive_minimum: None,
                exclusive_maximum: None
            }
        );
        assert!(s.params[0].required);
    }

    #[test]
    fn nullable_forms_are_normalized() {
        let s = spec(json!({
            "properties": {
                "a": { "type": ["integer", "null"] },
                "b": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                "c": { "oneOf": [{ "const": 1 }, { "const": 2 }] },
                "d": { "enum": ["x", null] }
            }
        }));
        assert_eq!(s.support, Support::Full);
        for name in ["a", "b", "d"] {
            assert!(s.params.iter().find(|p| p.name == name).unwrap().nullable);
        }
        assert_eq!(
            kind(&s, "c"),
            &ParamKind::Enum {
                values: vec![json!(1), json!(2)]
            }
        );
    }

    #[test]
    fn local_refs_are_resolved_and_enum_arrays_become_sets() {
        let s = spec(json!({
            "$defs": { "Color": { "enum": ["red", "blue"] } },
            "properties": {
                "colors": {
                    "type": "array",
                    "items": { "$ref": "#/$defs/Color" },
                    "maxItems": 2,
                    "uniqueItems": true
                }
            }
        }));
        assert_eq!(s.support, Support::Full);
        assert_eq!(
            kind(&s, "colors"),
            &ParamKind::EnumSet {
                values: vec![json!("red"), json!("blue")],
                min_items: None,
                max_items: Some(2),
                unique: true
            }
        );
    }

    #[test]
    fn unsupported_optional_param_makes_tool_partial() {
        let s = spec(json!({
            "properties": {
                "query": { "type": "string" },
                "passengers": { "type": "object", "properties": {} }
            },
            "required": ["query"]
        }));
        assert_eq!(
            s.support,
            Support::Partial(vec!["parameter `passengers`: nested object".into()])
        );
        assert!(s.is_routable());
        let strict = s.strict_args_schema();
        assert!(strict["properties"].get("passengers").is_none());
    }

    #[test]
    fn unsupported_required_param_or_top_level_blocks_the_tool() {
        let s = spec(json!({
            "properties": { "dates": { "allOf": [{ "type": "string" }] } },
            "required": ["dates"]
        }));
        assert!(!s.is_routable());

        let s = spec(json!({ "properties": {}, "oneOf": [{ "required": ["a"] }] }));
        assert_eq!(
            s.support,
            Support::Unsupported(vec!["top-level `oneOf` is not supported".into()])
        );

        let s = spec(
            json!({ "properties": { "x": { "$ref": "https://evil.test/s.json" } }, "required": ["x"] }),
        );
        assert_eq!(
            s.support,
            Support::Unsupported(vec![
                "parameter `x`: external $ref `https://evil.test/s.json`".into()
            ])
        );
    }

    #[test]
    fn strict_schema_requires_everything_and_nulls_optionals() {
        let s = spec(json!({
            "properties": {
                "category": { "type": "string", "enum": ["ALL", "SOCCER"] },
                "max_price": { "type": "number", "minimum": 0 },
                "query": { "type": "string" }
            },
            "required": ["query"]
        }));
        assert_eq!(
            s.strict_args_schema(),
            json!({
                "type": "object",
                "properties": {
                    "category": { "anyOf": [
                        { "type": "string", "enum": ["ALL", "SOCCER"] },
                        { "type": "null" }
                    ] },
                    "max_price": { "anyOf": [
                        { "type": "number", "description": ">= 0" },
                        { "type": "null" }
                    ] },
                    "query": { "type": "string" }
                },
                "required": ["category", "max_price", "query"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn arguments_are_validated_against_the_site_schema() {
        let s = spec(json!({
            "properties": {
                "category": { "type": "string", "enum": ["ALL", "SOCCER"] },
                "max_price": { "type": "number", "minimum": 0 }
            },
            "required": ["category"]
        }));
        assert!(
            s.validate_args(&json!({ "category": "SOCCER", "max_price": 10 }))
                .is_ok()
        );
        let errors = s
            .validate_args(&json!({ "category": "soccer", "max_price": -1 }))
            .unwrap_err();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(s.validate_args(&json!({})).is_err());
    }

    #[test]
    fn schema_hash_ignores_key_order() {
        let a = json!({ "type": "object", "properties": { "q": { "type": "string" } } });
        let b = json!({ "properties": { "q": { "type": "string" } }, "type": "object" });
        assert_eq!(schema_hash(&a), schema_hash(&b));
        assert_ne!(schema_hash(&a), schema_hash(&json!({ "type": "object" })));
        assert!(schema_hash(&a).starts_with("sha256:"));
    }
}
