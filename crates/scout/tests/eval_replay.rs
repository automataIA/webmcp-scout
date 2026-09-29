//! The committed eval datasets load cleanly, and replaying the recorded LLM
//! (and local model) responses reproduces the committed reports byte for byte.
//! `evals/` is not published with the repository, so these tests build only
//! with the `evals` feature: `cargo test -p scout --features evals --test eval_replay`.

use std::path::{Path, PathBuf};

use scout::eval::dataset::{Dataset, Split};
use scout::eval::{self, report};
use scout::llm::{CacheMode, DEFAULT_BASE_URL, DEFAULT_MODEL, LlmClient, LlmConfig};
use scout::local::{LocalClient, LocalConfig};
use scout::router::Router;
use scout::router::baseline::BaselineRouter;
use scout::router::hybrid::HybridRouter;
use scout::router::screen::screen;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dataset(scenarios: &str) -> Dataset {
    Dataset::load(&repo().join(scenarios), &repo().join("evals/toolsets")).unwrap()
}

#[test]
fn dataset_is_consistent() {
    let dataset = dataset("evals/scenarios");
    assert_eq!(dataset.scenarios.len(), 37);
    let calibration = dataset
        .scenarios
        .iter()
        .filter(|s| s.split == Split::Calibration)
        .count();
    assert_eq!(calibration, 16);
    assert_eq!(
        dataset.toolsets.keys().collect::<Vec<_>>(),
        ["alpha", "gamma", "shoes", "sport"]
    );
}

#[test]
fn adversarial_dataset_is_held_out() {
    let dataset = dataset("evals/adversarial");
    assert_eq!(dataset.scenarios.len(), 16);
    assert!(dataset.scenarios.iter().all(|s| s.split == Split::Test));
    assert_eq!(
        dataset.toolsets.keys().collect::<Vec<_>>(),
        ["adversarial", "shoes-injected", "sport-injected"]
    );
}

#[test]
fn sealed_dataset_is_held_out() {
    let dataset = dataset("evals/sealed");
    assert_eq!(dataset.scenarios.len(), 35);
    assert!(dataset.scenarios.iter().all(|s| s.split == Split::Test));
    assert!(dataset.toolsets.keys().all(|k| k.starts_with("sealed-")));
}

#[test]
fn training_dataset_is_disjoint_from_the_scored_sets() {
    let train = dataset("evals/train");
    assert!(train.scenarios.len() >= 400);
    assert!(train.scenarios.iter().all(|s| s.split == Split::Train));
    assert!(
        train
            .scenarios
            .iter()
            .filter(|s| s.tags.iter().any(|t| t == "missing-required"))
            .all(|s| s.intended_tool.is_some())
    );
    let norm = |r: &str| r.trim().to_lowercase();
    for scored in ["evals/scenarios", "evals/adversarial", "evals/sealed"] {
        for s in &dataset(scored).scenarios {
            let request = norm(&s.request);
            assert!(
                train.scenarios.iter().all(|t| norm(&t.request) != request),
                "{} is in the training set",
                s.id
            );
        }
    }
}

#[test]
fn clean_toolsets_pass_the_metadata_screen() {
    for (name, tools) in &dataset("evals/scenarios").toolsets {
        assert_eq!(screen(tools), None, "{name}");
    }
}

#[test]
fn injected_twins_keep_their_clean_labels() {
    let train = dataset("evals/train");
    let twins = Dataset::load(
        &repo().join("evals/train-twins"),
        &repo().join("evals/train-twins/toolsets"),
    )
    .unwrap();
    assert_eq!(twins.scenarios.len(), 2 * train.scenarios.len());
    for twin in &twins.scenarios {
        let stem = twin.id.rsplit_once('-').unwrap().0;
        let clean = train.scenarios.iter().find(|s| s.id == stem).unwrap();
        assert_eq!(
            (&twin.request, &twin.tool, &twin.args, &twin.absent),
            (&clean.request, &clean.tool, &clean.args, &clean.absent),
            "{}",
            twin.id
        );
        assert_eq!(twin.intended_tool, clean.intended_tool);
        assert!(twin.tags.contains(&format!("base-{}", clean.toolset)));
    }
}

#[tokio::test]
async fn replay_reproduces_the_committed_report() {
    assert_replay("evals/scenarios", "routing-baseline-v0").await;
}

#[tokio::test]
async fn replay_reproduces_the_committed_adversarial_report() {
    assert_replay("evals/adversarial", "routing-adversarial-baseline").await;
}

#[tokio::test]
async fn replay_reproduces_the_committed_local_adversarial_report() {
    assert_replay_with(
        "evals/adversarial",
        "routing-adversarial-local",
        Some("qwen3-4b-lora-t13"),
        false,
    )
    .await;
}

#[tokio::test]
async fn replay_reproduces_the_committed_adversarial_reports_of_the_untrusted_prompt_model() {
    for (name, screen) in [
        ("routing-adversarial-local-u1", false),
        ("routing-adversarial-local-u1-screen", true),
    ] {
        assert_replay_with("evals/adversarial", name, Some("qwen3-4b-lora-u1"), screen).await;
    }
}

#[tokio::test]
async fn replay_reproduces_the_committed_sealed_reports() {
    assert_replay("evals/sealed", "routing-sealed-baseline").await;
    for (name, screen) in [
        ("routing-sealed-local", false),
        ("routing-sealed-local-screen", true),
    ] {
        assert_replay_with("evals/sealed", name, Some("qwen3-4b-lora-u1"), screen).await;
    }
}

async fn assert_replay(scenarios: &str, name: &str) {
    assert_replay_with(scenarios, name, None, false).await;
}

/// With a local model name, replays the local router (the fan-out answers
/// from `evals/cache/local`), with the metadata screen if `screen_metadata`;
/// otherwise the baseline.
async fn assert_replay_with(
    scenarios: &str,
    name: &str,
    local_model: Option<&str>,
    screen_metadata: bool,
) {
    let llm = LlmClient::new(LlmConfig {
        base_url: DEFAULT_BASE_URL.into(),
        api_key: None,
        model: DEFAULT_MODEL.into(),
        reasoning_effort: Some("low".into()),
    })
    .unwrap()
    .with_cache(repo().join("evals/cache/llm"), CacheMode::Replay);
    let (meta, router) = match local_model {
        None => (
            report::baseline_meta(name, llm.model()),
            Router::Baseline(BaselineRouter::new(llm)),
        ),
        Some(model) => {
            let mut meta = report::local_meta(name, llm.model(), model);
            let local = LocalClient::new(LocalConfig {
                base_url: "http://127.0.0.1:9".into(),
                model: Some(model.into()),
            })
            .unwrap()
            .with_cache(repo().join("evals/cache/local"), CacheMode::Replay);
            let mut router = HybridRouter::local(local, BaselineRouter::new(llm));
            if screen_metadata {
                meta.router.push_str(
                    ", escalated to the baseline when tool metadata reads as an instruction",
                );
                router = router.with_metadata_screen();
            }
            (meta, Router::Hybrid(router))
        }
    };

    let results = eval::run_routing(&dataset(scenarios), &router, |_| {})
        .await
        .unwrap();
    let docs = repo().join("docs/evals");
    let md = std::fs::read_to_string(docs.join(format!("{name}.md"))).unwrap();
    let jsonl = std::fs::read_to_string(docs.join(format!("{name}.jsonl"))).unwrap();
    assert_eq!(report::render_markdown(&meta, &results), md);
    assert_eq!(report::render_jsonl(&results), jsonl);
}

#[test]
fn every_scenario_has_a_fanout_question_set() {
    for dir in ["evals/scenarios", "evals/adversarial"] {
        let dataset = dataset(dir);
        let records = eval::question_records(&dataset);
        assert_eq!(records.len(), dataset.scenarios.len());
        for r in &records {
            let set = r.question_set.as_ref().unwrap();
            assert_eq!(set.state["user_request"], r.scenario.request.as_str());
            assert!(
                set.questions.contains_key("__act__") && set.questions.contains_key("__tool__")
            );
        }
    }
}
