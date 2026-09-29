//! Shared helpers for browser integration tests: a static server for
//! `fixtures/sites` on an ephemeral localhost port.

use std::net::SocketAddr;
use std::path::PathBuf;

use tower_http::services::ServeDir;

pub struct FixtureServer {
    addr: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl FixtureServer {
    pub async fn start() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/sites");
        let app = axum::Router::new().fallback_service(ServeDir::new(root));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture server");
        let addr = listener.local_addr().expect("fixture server address");
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("fixture server");
        });
        Self { addr, task }
    }

    /// URL of a fixture site, e.g. `url("echo/")`.
    pub fn url(&self, path: &str) -> String {
        format!("http://{}/{path}", self.addr)
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
