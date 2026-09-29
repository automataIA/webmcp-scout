//! Trace viewer backend. It lists and reads the JSONL traces of a scout
//! project directory and starts `scout ask` runs as child processes. It
//! never drives Chrome itself: the `scout` CLI owns the browser and its
//! cleanup.
//!
//! The project directory is `SCOUT_DIR`, else the nearest ancestor of the
//! working directory holding `sites.toml`, else the working directory. The
//! CLI is `SCOUT_BIN`, else `target/{release,debug}/scout` in the project,
//! else `scout` on `PATH`.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use scout_types::{RunFinished, TraceFile};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

struct Project {
    dir: PathBuf,
    /// Process id of the `scout ask` in progress.
    run: Mutex<Option<u32>>,
}

impl Project {
    fn traces(&self) -> PathBuf {
        self.dir.join("traces")
    }
}

fn project_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SCOUT_DIR") {
        return dir.into();
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    cwd.ancestors()
        .find(|dir| dir.join("sites.toml").is_file())
        .map_or(cwd.clone(), Path::to_path_buf)
}

fn scout_bin(dir: &Path) -> PathBuf {
    if let Some(bin) = std::env::var_os("SCOUT_BIN") {
        return bin.into();
    }
    ["release", "debug"]
        .iter()
        .map(|profile| dir.join("target").join(profile).join("scout"))
        .find(|bin| bin.is_file())
        .unwrap_or_else(|| "scout".into())
}

/// `.jsonl` files of `dir`, newest first; a missing directory is empty.
fn list(dir: &Path) -> Result<Vec<TraceFile>, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    let mut files: Vec<TraceFile> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let meta = entry.metadata().ok()?;
            (name.ends_with(".jsonl") && meta.is_file()).then(|| TraceFile {
                name,
                bytes: meta.len(),
                modified: meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs()),
            })
        })
        .collect();
    files.sort_by(|a, b| {
        b.modified
            .cmp(&a.modified)
            .then_with(|| b.name.cmp(&a.name))
    });
    Ok(files)
}

/// `name` inside `dir`, when it is a plain `.jsonl` file name.
fn trace_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let plain = Path::new(name).file_name().is_some_and(|n| n == name);
    if !plain || name.starts_with('.') || !name.ends_with(".jsonl") {
        return Err(format!("not a trace file name: {name:?}"));
    }
    Ok(dir.join(name))
}

#[tauri::command]
fn list_traces(project: State<'_, Project>) -> Result<Vec<TraceFile>, String> {
    list(&project.traces())
}

#[tauri::command]
fn read_trace(project: State<'_, Project>, name: String) -> Result<String, String> {
    let path = trace_path(&project.traces(), &name)?;
    std::fs::read_to_string(&path).map_err(|e| format!("{name}: {e}"))
}

/// Starts `scout ask` on every configured site. Output lines arrive as
/// `run-log` events and the end as a `run-finished` event; returns the
/// trace file name.
#[tauri::command]
fn start_run(
    app: AppHandle,
    project: State<'_, Project>,
    request: String,
) -> Result<String, String> {
    let request = request.trim();
    if request.is_empty() {
        return Err("empty request".into());
    }
    let mut run = project.run.lock().map_err(|e| e.to_string())?;
    if run.is_some() {
        return Err("a run is already in progress".into());
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let trace = format!("{now}-viewer.jsonl");
    let mut child = Command::new(scout_bin(&project.dir))
        .arg("ask")
        .arg("--trace")
        .arg(Path::new("traces").join(&trace))
        // The request is data, never an option.
        .arg("--")
        .arg(request)
        .current_dir(&project.dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start scout: {e}"))?;
    *run = Some(child.id());

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let out_thread = forward(app.clone(), stdout);
    let err_thread = forward(app.clone(), stderr);
    let name = trace.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = out_thread.join();
        let _ = err_thread.join();
        if let Ok(mut run) = app.state::<Project>().run.lock() {
            *run = None;
        }
        let finished = match status {
            Ok(status) => RunFinished {
                trace: name,
                success: status.success(),
                code: status.code(),
            },
            Err(_) => RunFinished {
                trace: name,
                success: false,
                code: None,
            },
        };
        let _ = app.emit("run-finished", finished);
    });
    Ok(trace)
}

/// Emits every line of `output` as a `run-log` event.
fn forward(app: AppHandle, output: impl Read + Send + 'static) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        for line in BufReader::new(output).lines().map_while(Result::ok) {
            let _ = app.emit("run-log", line);
        }
    })
}

/// Asks a run still in progress to stop: `scout` handles SIGTERM by shutting
/// Chrome down and removing its profile.
fn stop_run(project: &Project) {
    let Some(pid) = project.run.lock().ok().and_then(|run| *run) else {
        return;
    };
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Project {
            dir: project_dir(),
            run: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![list_traces, read_trace, start_run])
        .build(tauri::generate_context!())
        .expect("error while building the viewer")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_run(&app.state::<Project>());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_jsonl_names_resolve() {
        let dir = Path::new("/p/traces");
        assert_eq!(
            trace_path(dir, "1-a+b.jsonl").unwrap(),
            Path::new("/p/traces/1-a+b.jsonl")
        );
        for bad in [
            "../x.jsonl",
            "a/b.jsonl",
            "/etc/x.jsonl",
            ".hidden.jsonl",
            "x.json",
            "..",
            "",
        ] {
            assert!(trace_path(dir, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn lists_traces_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        for (name, secs) in [("old.jsonl", 100), ("new.jsonl", 200), ("notes.txt", 300)] {
            let path = dir.path().join(name);
            std::fs::write(&path, "{}\n").unwrap();
            let time = UNIX_EPOCH + std::time::Duration::from_secs(secs);
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(time)
                .unwrap();
        }
        std::fs::create_dir(dir.path().join("dir.jsonl")).unwrap();
        let names: Vec<String> = list(dir.path())
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();
        assert_eq!(names, ["new.jsonl", "old.jsonl"]);
        assert!(list(&dir.path().join("missing")).unwrap().is_empty());
    }

    #[test]
    fn csp_lets_the_ui_fetch_its_own_wasm() {
        // Trunk's loader fetches the `.wasm`, which falls under `connect-src`
        // (not `default-src`); without `'self'` the window stays blank.
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        for key in ["csp", "devCsp"] {
            let csp = &conf["app"]["security"][key];
            let connect = csp["connect-src"].as_str().unwrap();
            assert!(
                connect.split(' ').any(|s| s == "'self'"),
                "{key}: {connect}"
            );
            let script = csp["script-src"].as_str().unwrap();
            assert!(script.contains("'wasm-unsafe-eval'"), "{key}: {script}");
        }
    }
}
