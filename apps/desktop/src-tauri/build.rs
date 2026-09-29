fn main() {
    // App commands get `allow-*` permissions and are denied unless a
    // capability grants them.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&["list_traces", "read_trace", "start_run"]),
    ))
    .expect("tauri build script");
}
