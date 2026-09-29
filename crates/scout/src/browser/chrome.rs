//! Locating the Chrome binary and the command-line flags it is launched with.

use std::path::{Path, PathBuf};

/// Flags passed to Chrome. chromiumoxide's defaults are disabled because they
/// carry their own `--enable-features`, which would conflict with ours, and
/// `--enable-automation`. `.arg()` adds the leading `--` itself.
pub(super) const CHROME_ARGS: &[&str] = &[
    // WebMCP on origins without an origin-trial token (localhost fixtures,
    // legacy demos). Verified on Chrome 154.
    "enable-features=WebMCPTesting",
    "disable-features=TranslateUI",
    "no-first-run",
    "no-default-browser-check",
    "password-store=basic",
    "use-mock-keychain",
    "disable-background-networking",
    "disable-background-timer-throttling",
    "disable-backgrounding-occluded-windows",
    "disable-renderer-backgrounding",
    "disable-breakpad",
    "disable-client-side-phishing-detection",
    "disable-default-apps",
    "disable-dev-shm-usage",
    "disable-hang-monitor",
    "disable-popup-blocking",
    "disable-prompt-on-repost",
    "disable-sync",
    "metrics-recording-only",
    "lang=en-US",
];

/// Finds a Chrome/Chromium binary: `SCOUT_CHROME`, then the usual names on
/// `PATH`, then the newest Playwright Chromium build.
pub fn resolve_chrome() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SCOUT_CHROME").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(path));
    }
    [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    .iter()
    .find_map(|name| find_in_path(name))
    .or_else(playwright_chromium)
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

fn playwright_chromium() -> Option<PathBuf> {
    let root = Path::new(&std::env::var_os("HOME")?).join(".cache/ms-playwright");
    let mut builds: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("chromium-"))
        })
        .collect();
    builds.sort();
    builds.iter().rev().find_map(|dir| {
        ["chrome-linux64/chrome", "chrome-linux/chrome"]
            .iter()
            .map(|rel| dir.join(rel))
            .find(|p| p.is_file())
    })
}
