//! `scout` stops cleanly when interrupted: no Chrome process
//! outlives it and its temporary profile is removed. Linux only (`/proc`).

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

fn wait_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while !done() {
        if Instant::now() >= deadline {
            return false;
        }
        sleep(Duration::from_millis(100));
    }
    true
}

/// Arguments of `pid`. Chrome rewrites its `argv` into one space-separated
/// string, so spaces separate arguments too.
fn cmdline(pid: &str) -> Vec<String> {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|raw| {
            raw.split(|&b| b == 0 || b == b' ')
                .map(|arg| String::from_utf8_lossy(arg).into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Parent pid from `/proc/<pid>/stat`, read after the `(comm)` field.
fn parent(pid: &str) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(1).map(str::to_owned)
}

fn pids() -> Vec<String> {
    std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.bytes().all(|b| b.is_ascii_digit()))
        .collect()
}

/// The `--user-data-dir=...` argument of the Chrome launched by `pid`.
fn profile_flag(pid: u32) -> Option<String> {
    let pid = pid.to_string();
    pids()
        .into_iter()
        .filter(|p| parent(p).as_deref() == Some(pid.as_str()))
        .flat_map(|p| cmdline(&p))
        .find(|arg| arg.starts_with("--user-data-dir="))
}

fn profile_users(flag: &str) -> usize {
    pids()
        .iter()
        .filter(|p| cmdline(p).iter().any(|arg| arg == flag))
        .count()
}

/// Kills the whole group of a test gone wrong, Chrome included.
fn kill_group(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .status();
}

/// Starts `scout` waiting on a page that never registers a tool, sends the
/// signal `settle` after Chrome started, and checks the cleanup.
fn interrupt(kill_args: impl FnOnce(u32) -> Vec<String>, signal: &str, settle: Duration) {
    let mut scout = Command::new(env!("CARGO_BIN_EXE_scout"))
        .args(["tools", "about:blank", "--wait", "60"])
        // Its own process group, like a job started from a terminal.
        .process_group(0)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start scout");
    let pid = scout.id();

    let mut flag = None;
    let launched = wait_until(Duration::from_secs(30), || {
        flag = profile_flag(pid);
        flag.is_some()
    });
    if !launched {
        kill_group(pid);
        panic!("Chrome did not start");
    }
    let flag = flag.unwrap();
    let profile = flag.trim_start_matches("--user-data-dir=").to_owned();
    sleep(settle);

    let sent = Command::new("kill").args(kill_args(pid)).status().unwrap();
    assert!(sent.success(), "kill failed");
    let mut status: Option<ExitStatus> = None;
    let exited = wait_until(Duration::from_secs(20), || {
        status = scout.try_wait().unwrap();
        status.is_some()
    });
    if !exited {
        kill_group(pid);
        panic!("scout did not exit after {signal}");
    }
    let mut stderr = String::new();
    scout
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();

    assert!(!status.unwrap().success());
    assert!(
        stderr.contains(&format!("interrupted by {signal}")),
        "stderr: {stderr}"
    );
    assert_eq!(profile_users(&flag), 0, "Chrome outlived scout");
    assert!(
        !Path::new(&profile).exists(),
        "profile {profile} left behind"
    );
}

/// Once the tab is open, so that the signal lands mid-command.
const MID_COMMAND: Duration = Duration::from_secs(1);

#[test]
fn ctrl_c_on_the_terminal_group_leaves_nothing_behind() {
    // A terminal Ctrl-C reaches every process of the foreground group,
    // Chrome included.
    interrupt(
        |pid| vec!["-INT".into(), "--".into(), format!("-{pid}")],
        "SIGINT",
        MID_COMMAND,
    );
}

#[test]
fn sigterm_to_scout_alone_leaves_nothing_behind() {
    interrupt(
        |pid| vec!["-TERM".into(), pid.to_string()],
        "SIGTERM",
        MID_COMMAND,
    );
}

#[test]
fn sigterm_while_chrome_launches_leaves_nothing_behind() {
    interrupt(
        |pid| vec!["-TERM".into(), pid.to_string()],
        "SIGTERM",
        Duration::ZERO,
    );
}
