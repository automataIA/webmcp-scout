//! Chrome driven over CDP (chromiumoxide). One tab per site; WebMCP discovery
//! and execution run as small JS functions called with
//! `Runtime.callFunctionOn`, arguments passed as CDP values.

mod chrome;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chromiumoxide::cdp::js_protocol::runtime::{CallArgument, CallFunctionOnParams};
use chromiumoxide::error::CdpError;
use chromiumoxide::handler::viewport::Viewport;
use chromiumoxide::{Browser, BrowserConfig, Page};
use futures::StreamExt;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::task::JoinHandle;

pub use chrome::resolve_chrome;

use crate::webmcp::{DiscoveredTool, Discovery, RejectedTool, parse_input_schema};

const DISCOVER_JS: &str = include_str!("js/discover.js");
const EXECUTE_JS: &str = include_str!("js/execute.js");
const LEGACY_SHIM_JS: &str = include_str!("js/legacy_shim.js");

/// How long `Browser.close` and process exit may take before Chrome is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const PROFILE_POLL_INTERVAL: Duration = Duration::from_millis(50);
const TOOLS_SETTLE: Duration = Duration::from_millis(600);
const TOOL_POLL_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, thiserror::Error)]
pub enum BrowserError {
    #[error("no Chrome/Chromium executable found (set SCOUT_CHROME)")]
    ChromeNotFound,
    #[error("failed to launch Chrome: {0}")]
    Launch(String),
    #[error("the CDP connection to Chrome is closed")]
    Disconnected,
    #[error("WebMCP (document.modelContext) is not available on {url}")]
    WebMcpUnsupported { url: String },
    #[error("tool `{0}` is not registered on the current page")]
    ToolNotFound(String),
    #[error("tool `{tool}` failed: {message}")]
    ToolFailed { tool: String, message: String },
    #[error("the page navigated and its JS context was destroyed during the call")]
    ContextDestroyed,
    #[error("{0} timed out after {1:?}")]
    Timeout(&'static str, Duration),
    #[error("unexpected bridge response: {0}")]
    Bridge(String),
    #[error("CDP error: {0}")]
    Cdp(String),
}

fn from_cdp(err: CdpError) -> BrowserError {
    const DESTROYED: [&str; 3] = [
        "Execution context was destroyed",
        "Cannot find context with specified id",
        "Inspected target navigated or closed",
    ];
    let msg = err.to_string();
    if DESTROYED.iter().any(|m| msg.contains(m)) {
        BrowserError::ContextDestroyed
    } else {
        BrowserError::Cdp(msg)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BrowserOptions {
    /// Chrome binary; resolved with [`resolve_chrome`] when `None`.
    pub executable: Option<PathBuf>,
    /// Show the browser window instead of running headless.
    pub headed: bool,
}

/// A Chrome process with a throwaway profile, owned by this runtime.
pub struct ChromeBrowser {
    browser: Browser,
    alive: Arc<AtomicBool>,
    handler: JoinHandle<()>,
    version: String,
    // Declared last so that, if `shutdown` is skipped, the profile directory
    // is removed after Chrome is gone.
    profile: TempDir,
}

impl ChromeBrowser {
    pub async fn launch(opts: &BrowserOptions) -> Result<Self, BrowserError> {
        let executable = match &opts.executable {
            Some(path) => path.clone(),
            None => resolve_chrome().ok_or(BrowserError::ChromeNotFound)?,
        };
        let profile = tempfile::Builder::new()
            .prefix("webmcp-scout-profile-")
            .tempdir()
            .map_err(|e| BrowserError::Launch(format!("temporary profile: {e}")))?;

        let builder = BrowserConfig::builder()
            .chrome_executable(&executable)
            .user_data_dir(profile.path())
            .window_size(1280, 900)
            .viewport(None::<Viewport>)
            .disable_default_args()
            .args(chrome::CHROME_ARGS.iter().copied());
        let builder = if opts.headed {
            builder.with_head()
        } else {
            builder.new_headless_mode()
        };
        let config = builder.build().map_err(BrowserError::Launch)?;

        let (browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| BrowserError::Launch(e.to_string()))?;

        // The handler drives the CDP connection and must be polled for the
        // whole browser lifetime; when it ends, every later call fails fast.
        let alive = Arc::new(AtomicBool::new(true));
        let flag = Arc::clone(&alive);
        let handler = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if let Err(e) = event {
                    tracing::debug!(error = %e, "CDP handler error");
                }
            }
            flag.store(false, Ordering::SeqCst);
            tracing::debug!("CDP handler stopped");
        });

        let version = browser.version().await.map_err(from_cdp)?.product;
        tracing::info!(%version, executable = %executable.display(), "Chrome launched");

        Ok(Self {
            browser,
            alive,
            handler,
            version,
            profile,
        })
    }

    /// Product string reported by Chrome, e.g. `HeadlessChrome/154.0.8037.57`.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Opens `url` in a new tab. With `legacy_shim`, the compatibility shim for
    /// the old `navigator.modelContext` API is installed before page scripts.
    pub async fn open(&self, url: &str, opts: &TabOptions) -> Result<Tab, BrowserError> {
        ensure_alive(&self.alive)?;
        let page = self
            .browser
            .new_page("about:blank")
            .await
            .map_err(from_cdp)?;
        if opts.legacy_shim {
            page.add_init_script(LEGACY_SHIM_JS)
                .await
                .map_err(from_cdp)?;
        }
        page.goto(url).await.map_err(from_cdp)?;
        Ok(Tab {
            page,
            alive: Arc::clone(&self.alive),
        })
    }

    /// Closes Chrome, killing it if it does not exit within the grace period.
    pub async fn shutdown(self) {
        let Self {
            mut browser,
            handler,
            profile,
            ..
        } = self;
        let closed = matches!(
            tokio::time::timeout(SHUTDOWN_GRACE, browser.close()).await,
            Ok(Ok(_))
        );
        let exited = closed
            && matches!(
                tokio::time::timeout(SHUTDOWN_GRACE, browser.wait()).await,
                Ok(Ok(_))
            );
        if !exited {
            tracing::warn!("Chrome did not exit cleanly, killing it");
            browser.kill().await;
        }
        handler.abort();
        remove_profile(profile).await;
    }
}

/// Chrome helper processes (network service, zygotes) can outlive the browser
/// process and keep writing to the profile while they flush. Wait until no
/// process runs with our `--user-data-dir`, then delete the profile.
async fn remove_profile(profile: TempDir) {
    let flag = format!("--user-data-dir={}", profile.path().display());
    let deadline = Instant::now() + SHUTDOWN_GRACE;
    while profile_in_use(&flag) {
        if Instant::now() >= deadline {
            tracing::warn!("Chrome helper processes still running, removing the profile anyway");
            break;
        }
        tokio::time::sleep(PROFILE_POLL_INTERVAL).await;
    }
    let path = profile.path().to_owned();
    if let Err(e) = profile.close() {
        tracing::warn!(path = %path.display(), error = %e, "could not remove the temporary Chrome profile");
    }
}

/// True if a process has `flag` among its arguments. Linux only (`/proc`);
/// elsewhere nothing is found and the profile is removed right away.
fn profile_in_use(flag: &str) -> bool {
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return false;
    };
    procs.filter_map(Result::ok).any(|entry| {
        std::fs::read(entry.path().join("cmdline")).is_ok_and(|cmdline| has_arg(&cmdline, flag))
    })
}

/// Chrome rewrites its `argv` into one space-separated string, so the
/// arguments are split on spaces as well as NULs.
fn has_arg(cmdline: &[u8], flag: &str) -> bool {
    cmdline
        .split(|&b| b == 0 || b == b' ')
        .any(|arg| arg == flag.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_profile_flag_in_rewritten_and_plain_cmdlines() {
        let flag = "--user-data-dir=/tmp/webmcp-scout-profile-abc";
        let chrome = b"/opt/google/chrome/chrome --type=renderer --user-data-dir=/tmp/webmcp-scout-profile-abc --lang=en\0";
        let plain = b"chrome\0--user-data-dir=/tmp/webmcp-scout-profile-abc\0";
        let other = b"chrome --user-data-dir=/tmp/webmcp-scout-profile-abcd\0";
        assert!(has_arg(chrome, flag));
        assert!(has_arg(plain, flag));
        assert!(!has_arg(other, flag));
    }
}

fn ensure_alive(alive: &AtomicBool) -> Result<(), BrowserError> {
    if alive.load(Ordering::SeqCst) {
        Ok(())
    } else {
        Err(BrowserError::Disconnected)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TabOptions {
    pub legacy_shim: bool,
}

/// How tool arguments are handed to `document.modelContext.executeTool`.
/// The draft spec takes an object; Chrome 154 also accepts a JSON string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgsEncoding {
    Object,
    JsonString,
}

impl ArgsEncoding {
    fn as_str(self) -> &'static str {
        match self {
            Self::Object => "object",
            Self::JsonString => "json_string",
        }
    }
}

/// Encoding used by [`Tab::execute`]. Chrome 154 rejects an object with
/// "Failed to parse input arguments" and requires a JSON string.
pub const DEFAULT_ARGS_ENCODING: ArgsEncoding = ArgsEncoding::JsonString;

/// Outcome of one tool call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolOutcome {
    pub tool: String,
    /// Result as returned by the page. Tools usually return strings (plain
    /// text or JSON); other values are JSON-encoded by the bridge.
    pub output: Option<String>,
    pub url_before: String,
    pub url_after: String,
    pub elapsed_ms: u64,
}

impl ToolOutcome {
    /// True when the call changed the page URL (tools may be contextual, so
    /// callers should rediscover).
    pub fn navigated(&self) -> bool {
        self.url_before != self.url_after
    }
}

pub struct Tab {
    page: Page,
    alive: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct RawDiscovery {
    supported: bool,
    url: String,
    origin: String,
    tools: Vec<RawTool>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTool {
    name: String,
    title: Option<String>,
    description: String,
    input_schema: Option<String>,
    origin: String,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum RawExecution {
    Ok { result: Option<String>, url: String },
    NotFound,
    Unsupported { url: String },
    Error { error: String },
}

impl Tab {
    pub async fn url(&self) -> Result<String, BrowserError> {
        ensure_alive(&self.alive)?;
        Ok(self.page.url().await.map_err(from_cdp)?.unwrap_or_default())
    }

    /// Lists the tools of the current document. Tools from other origins
    /// (iframes) and tools with an unusable schema are reported as rejected.
    pub async fn discover(&self) -> Result<Discovery, BrowserError> {
        let raw: RawDiscovery = self.call(DISCOVER_JS, Vec::new()).await?;
        if !raw.supported {
            return Err(BrowserError::WebMcpUnsupported { url: raw.url });
        }
        let mut tools = Vec::new();
        let mut rejected = Vec::new();
        for tool in raw.tools {
            if tool.origin != raw.origin {
                rejected.push(RejectedTool {
                    name: tool.name,
                    reason: format!("registered by another origin ({})", tool.origin),
                });
                continue;
            }
            match parse_input_schema(tool.input_schema.as_deref()) {
                Ok(input_schema) => tools.push(DiscoveredTool {
                    name: tool.name,
                    title: tool.title,
                    description: tool.description,
                    input_schema,
                    origin: tool.origin,
                }),
                Err(reason) => rejected.push(RejectedTool {
                    name: tool.name,
                    reason,
                }),
            }
        }
        Ok(Discovery {
            url: raw.url,
            origin: raw.origin,
            tools,
            rejected,
        })
    }

    /// Polls [`Tab::discover`] until the page has registered tools and the set
    /// has stayed unchanged for a short settle window, or `timeout` elapses.
    /// Pages register tools asynchronously and sometimes in several batches.
    /// At the deadline returns the latest (possibly empty) discovery, or
    /// `WebMcpUnsupported` if the API never appeared.
    pub async fn wait_for_tools(&self, timeout: Duration) -> Result<Discovery, BrowserError> {
        let deadline = Instant::now() + timeout;
        // Last non-empty discovery and when its tool set was first seen.
        let mut stable: Option<(Discovery, Instant)> = None;
        loop {
            let result = self.discover().await;
            // Transient while the page is still navigating or starting up.
            let transient = matches!(
                result,
                Err(BrowserError::ContextDestroyed | BrowserError::WebMcpUnsupported { .. })
            );
            let now = Instant::now();
            match result {
                Ok(d) if !d.tools.is_empty() => {
                    let since = match &stable {
                        Some((prev, since)) if prev.tools == d.tools => *since,
                        _ => now,
                    };
                    if now >= deadline || now.duration_since(since) >= TOOLS_SETTLE {
                        return Ok(d);
                    }
                    stable = Some((d, since));
                }
                result if now >= deadline => return result,
                Err(e) if !transient => return Err(e),
                _ => stable = None,
            }
            tokio::time::sleep(TOOL_POLL_INTERVAL).await;
        }
    }

    /// Executes `tool` with `args` using [`DEFAULT_ARGS_ENCODING`].
    ///
    /// On timeout the tool may still be running inside the page.
    pub async fn execute(
        &self,
        tool: &str,
        args: &Value,
        timeout: Duration,
    ) -> Result<ToolOutcome, BrowserError> {
        self.execute_with_encoding(tool, args, DEFAULT_ARGS_ENCODING, timeout)
            .await
    }

    pub async fn execute_with_encoding(
        &self,
        tool: &str,
        args: &Value,
        encoding: ArgsEncoding,
        timeout: Duration,
    ) -> Result<ToolOutcome, BrowserError> {
        let url_before = self.url().await?;
        let started = Instant::now();
        let call = self.call::<RawExecution>(
            EXECUTE_JS,
            vec![json!(tool), args.clone(), json!(encoding.as_str())],
        );
        let raw = tokio::time::timeout(timeout, call)
            .await
            .map_err(|_| BrowserError::Timeout("tool call", timeout))??;
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        match raw {
            RawExecution::Ok { result, url } => Ok(ToolOutcome {
                tool: tool.to_owned(),
                output: result,
                url_before,
                url_after: url,
                elapsed_ms,
            }),
            RawExecution::NotFound => Err(BrowserError::ToolNotFound(tool.to_owned())),
            RawExecution::Unsupported { url } => Err(BrowserError::WebMcpUnsupported { url }),
            RawExecution::Error { error } => Err(BrowserError::ToolFailed {
                tool: tool.to_owned(),
                message: error,
            }),
        }
    }

    pub async fn close(self) -> Result<(), BrowserError> {
        ensure_alive(&self.alive)?;
        self.page.close().await.map_err(from_cdp)
    }

    async fn call<T: DeserializeOwned>(
        &self,
        function: &str,
        args: Vec<Value>,
    ) -> Result<T, BrowserError> {
        ensure_alive(&self.alive)?;
        let mut params = CallFunctionOnParams::builder()
            .function_declaration(function)
            .await_promise(true)
            .return_by_value(true);
        for arg in args {
            params = params.argument(CallArgument::builder().value(arg).build());
        }
        let params = params.build().map_err(BrowserError::Bridge)?;
        let result = self
            .page
            .evaluate_function(params)
            .await
            .map_err(from_cdp)?;
        result
            .into_value()
            .map_err(|e| BrowserError::Bridge(e.to_string()))
    }
}
