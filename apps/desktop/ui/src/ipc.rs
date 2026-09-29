//! Bridge to the Tauri backend. In a plain browser (`trunk serve`, UI tests)
//! `window.__TAURI__` is missing and every call falls back to a bundled
//! sample trace, so the UI runs and can be tested without the backend.

use scout_types::{RunFinished, TraceFile};
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// A real `scout ask` trace on the sport-shop and Shoe Store demos.
pub const SAMPLE_NAME: &str = "sample-sport+shoes.jsonl";
const SAMPLE: &str = include_str!("../sample.jsonl");

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], catch)]
    async fn invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"], catch)]
    async fn listen(event: &str, handler: &js_sys::Function) -> Result<JsValue, JsValue>;
}

pub fn is_tauri() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("__TAURI__"))
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

async fn call<A: Serialize, R: DeserializeOwned>(cmd: &str, args: &A) -> Result<R, String> {
    let args = serde_wasm_bindgen::to_value(args).map_err(|e| e.to_string())?;
    let out = invoke(cmd, args).await.map_err(|e| {
        e.as_string()
            .or_else(|| js_sys::JSON::stringify(&e).ok().map(String::from))
            .unwrap_or_else(|| "unknown IPC error".into())
    })?;
    serde_wasm_bindgen::from_value(out).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct NoArgs {}

pub async fn list_traces() -> Result<Vec<TraceFile>, String> {
    if !is_tauri() {
        return Ok(vec![TraceFile {
            name: SAMPLE_NAME.into(),
            bytes: SAMPLE.len() as u64,
            modified: 0,
        }]);
    }
    call("list_traces", &NoArgs {}).await
}

/// The JSONL text of a trace from the traces directory.
pub async fn read_trace(name: &str) -> Result<String, String> {
    if !is_tauri() {
        return Ok(SAMPLE.into());
    }
    #[derive(Serialize)]
    struct Args<'a> {
        name: &'a str,
    }
    call("read_trace", &Args { name }).await
}

/// Starts `scout ask` on every configured site; returns the trace name.
pub async fn start_run(request: &str) -> Result<String, String> {
    if !is_tauri() {
        return Err("runs need the desktop app".into());
    }
    #[derive(Serialize)]
    struct Args<'a> {
        request: &'a str,
    }
    call("start_run", &Args { request }).await
}

/// Calls `on_line` for each output line of the current run and `on_finished`
/// when it exits. Does nothing outside Tauri.
pub fn on_run_events(
    on_line: impl Fn(String) + 'static,
    on_finished: impl Fn(RunFinished) + 'static,
) {
    if !is_tauri() {
        return;
    }
    subscribe("run-log", on_line);
    subscribe("run-finished", on_finished);
}

fn subscribe<T: DeserializeOwned + 'static>(event: &'static str, callback: impl Fn(T) + 'static) {
    #[derive(serde::Deserialize)]
    struct Event<T> {
        payload: T,
    }
    let closure = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
        if let Ok(event) = serde_wasm_bindgen::from_value::<Event<T>>(event) {
            callback(event.payload);
        }
    });
    let handler: js_sys::Function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    // Listens for the whole life of the app.
    closure.forget();
    wasm_bindgen_futures::spawn_local(async move {
        let _ = listen(event, &handler).await;
    });
}

/// Text of a file the user picked in the browser's file input.
pub async fn read_file(file: web_sys::File) -> Result<String, String> {
    let text = wasm_bindgen_futures::JsFuture::from(file.text())
        .await
        .map_err(|e| format!("{e:?}"))?;
    text.as_string().ok_or_else(|| "file is not text".into())
}
