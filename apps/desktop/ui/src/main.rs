//! Trace viewer UI: lists and shows the JSONL traces of `scout ask` runs and
//! starts new runs through the Tauri backend.

mod app;
mod ipc;
mod summary;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::App);
}
