//! Record/replay response cache shared by the LLM and Jev clients: one JSON
//! file per request, named by the sha256 of the canonical request body, so
//! evals replay offline and deterministically.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    Off,
    /// Serve hits from the cache, call the API on misses and record them.
    Record,
    /// Serve only from the cache; a miss is an error. Needs no API key.
    Replay,
}

#[derive(Serialize, Deserialize)]
struct Entry<T> {
    request: Value,
    response: T,
}

fn path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.json"))
}

/// The recorded response for `key`, if any.
pub(crate) fn read<T: DeserializeOwned>(dir: &Path, key: &str) -> Result<Option<T>, String> {
    let path = path(dir, key);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let entry: Entry<T> =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Some(entry.response))
}

pub(crate) fn write<T: Serialize>(
    dir: &Path,
    key: &str,
    request: &Value,
    response: &T,
) -> Result<(), String> {
    let entry = Entry {
        request: request.clone(),
        response,
    };
    let text = serde_json::to_string_pretty(&entry).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(path(dir, key), text + "\n"))
        .map_err(|e| format!("{}: {e}", dir.display()))
}
