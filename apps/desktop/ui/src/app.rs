use std::sync::Arc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use scout_types::{Event, Record, TraceFile, parse_jsonl};
use web_sys::HtmlInputElement;

use crate::ipc;
use crate::summary::{self, SiteRun};

/// Output lines of the current run kept on screen.
const LOG_LINES: usize = 200;
/// Ranked products listed before "... more".
const TOP: usize = 20;

type Loaded = Option<Result<Arc<Vec<Record>>, String>>;

#[component]
pub fn App() -> impl IntoView {
    let traces = RwSignal::new(Vec::<TraceFile>::new());
    let selected = RwSignal::new(None::<String>);
    let loaded = RwSignal::new(Loaded::None);
    let notice = RwSignal::new(None::<String>);
    let request = RwSignal::new(String::new());
    let running = RwSignal::new(false);
    let log = RwSignal::new(Vec::<String>::new());

    let show = move |name: String, text: Result<String, String>| {
        selected.set(Some(name.clone()));
        let parsed = text.and_then(|t| parse_jsonl(&t).map_err(|e| format!("{name}: {e}")));
        loaded.set(Some(parsed.map(Arc::new)));
    };
    let open = move |name: String| {
        spawn_local(async move {
            let text = ipc::read_trace(&name).await;
            show(name, text);
        });
    };
    let refresh = move |then_open: Option<String>| {
        spawn_local(async move {
            match ipc::list_traces().await {
                Ok(list) => {
                    let first = list.first().map(|f| f.name.clone());
                    traces.set(list);
                    if let Some(name) =
                        then_open.or(first.filter(|_| selected.get_untracked().is_none()))
                    {
                        open(name);
                    }
                }
                Err(e) => notice.set(Some(e)),
            }
        });
    };
    refresh(None);

    ipc::on_run_events(
        move |line| {
            log.update(|log| {
                log.push(line);
                if log.len() > LOG_LINES {
                    log.remove(0);
                }
            })
        },
        move |finished| {
            running.set(false);
            if !finished.success {
                notice.set(Some(match finished.code {
                    Some(code) => format!("scout exited with code {code}"),
                    None => "scout was stopped by a signal".into(),
                }));
            }
            refresh(Some(finished.trace));
        },
    );

    let start = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        let text = request.get_untracked().trim().to_owned();
        if text.is_empty() || running.get_untracked() {
            return;
        }
        running.set(true);
        notice.set(None);
        log.set(Vec::new());
        spawn_local(async move {
            if let Err(e) = ipc::start_run(&text).await {
                running.set(false);
                notice.set(Some(e));
            }
        });
    };

    let pick_file = move |ev: leptos::ev::Event| {
        let input = event_target::<HtmlInputElement>(&ev);
        let Some(file) = input.files().and_then(|files| files.get(0)) else {
            return;
        };
        spawn_local(async move {
            let name = file.name();
            let text = ipc::read_file(file).await;
            show(name, text);
        });
    };

    view! {
        <div class="flex h-screen bg-base-200 text-base-content">
            <aside class="w-80 shrink-0 flex flex-col border-r border-base-300 bg-base-100">
                <header class="px-4 py-3 border-b border-base-300">
                    <h1 class="font-semibold">"WebMCP Scout"</h1>
                    <p class="text-xs opacity-60">"Run traces"</p>
                </header>
                <form class="p-4 space-y-2 border-b border-base-300" on:submit=start>
                    <input
                        class="input input-sm w-full"
                        placeholder="soccer boots under 150 dollars"
                        aria-label="Request"
                        bind:value=request
                        disabled=move || running.get() || !ipc::is_tauri()
                    />
                    <button
                        class="btn btn-primary btn-sm w-full"
                        type="submit"
                        disabled=move || running.get() || !ipc::is_tauri()
                        title=move || (!ipc::is_tauri()).then_some("Runs need the desktop app")
                    >
                        {move || if running.get() { "Running…" } else { "Ask every site" }}
                    </button>
                </form>
                <div class="px-4 pt-3 pb-1 flex items-center justify-between">
                    <span class="text-xs font-semibold uppercase opacity-60">"Traces"</span>
                    <button class="btn btn-ghost btn-xs" on:click=move |_| refresh(None)>
                        "Refresh"
                    </button>
                </div>
                <ul class="menu menu-sm w-full flex-1 overflow-y-auto flex-nowrap">
                    <For
                        each=move || traces.get()
                        key=|f| f.name.clone()
                        children=move |file| {
                            let name = file.name.clone();
                            let active = name.clone();
                            view! {
                                <li>
                                    <a
                                        class="flex flex-col items-start gap-0"
                                        class:menu-active=move || {
                                            selected.get().as_deref() == Some(active.as_str())
                                        }
                                        on:click=move |_| open(name.clone())
                                    >
                                        <span class="font-mono text-xs break-all">
                                            {file.name.clone()}
                                        </span>
                                        <span class="text-xs opacity-60 tnum">
                                            {format_modified(file.modified)} " · "
                                            {format_bytes(file.bytes)}
                                        </span>
                                    </a>
                                </li>
                            }
                        }
                    />
                </ul>
                <label class="p-4 border-t border-base-300 text-xs space-y-1">
                    <span class="opacity-60">"Open a trace file"</span>
                    <input
                        type="file"
                        accept=".jsonl,application/jsonl,text/plain"
                        class="file-input file-input-sm w-full"
                        on:change=pick_file
                    />
                </label>
            </aside>
            <main class="flex-1 min-w-0 overflow-y-auto p-6 space-y-6">
                {move || {
                    notice
                        .get()
                        .map(|text| {
                            view! {
                                <div role="alert" class="alert alert-error alert-soft">
                                    <span class="break-all">{text}</span>
                                    <button class="btn btn-ghost btn-xs" on:click=move |_| notice.set(None)>
                                        "Dismiss"
                                    </button>
                                </div>
                            }
                        })
                }}
                <Show when=move || running.get() || !log.get().is_empty()>
                    <section class="card bg-base-100 border border-base-300">
                        <div class="card-body p-4">
                            <h2 class="card-title text-sm">
                                {move || if running.get() { "Run in progress" } else { "Last run output" }}
                            </h2>
                            <pre class="text-xs font-mono max-h-48 overflow-y-auto whitespace-pre-wrap break-all">
                                {move || log.get().join("\n")}
                            </pre>
                        </div>
                    </section>
                </Show>
                {move || match loaded.get() {
                    None => view! { <p class="opacity-60">"No trace selected."</p> }.into_any(),
                    Some(Err(e)) => {
                        view! {
                            <div role="alert" class="alert alert-warning">
                                <span class="break-all">{e}</span>
                            </div>
                        }
                            .into_any()
                    }
                    Some(Ok(records)) => view! { <TraceView records /> }.into_any(),
                }}
            </main>
        </div>
    }
}

#[component]
fn TraceView(records: Arc<Vec<Record>>) -> impl IntoView {
    let site_filter = RwSignal::new(None::<String>);
    let open_row = RwSignal::new(None::<usize>);
    let request = summary::request(&records)
        .unwrap_or("(no request)")
        .to_owned();
    let duration = summary::duration_ms(&records);
    let sites = summary::sites(&records);
    let ranked = records.iter().find_map(|r| match &r.event {
        Event::Ranked { kept, excluded, .. } => Some((kept.clone(), excluded.clone())),
        _ => None,
    });
    let count = records.len();

    let rows = {
        let records = records.clone();
        move || {
            let filter = site_filter.get();
            records
                .iter()
                .enumerate()
                .filter(|(_, r)| filter.is_none() || r.site == filter)
                .map(|(i, r)| (i, r.clone()))
                .collect::<Vec<_>>()
        }
    };

    view! {
        <section class="space-y-1">
            <h2 class="text-xl font-semibold break-words">{request}</h2>
            <p class="text-sm opacity-60 tnum">
                {format!("{count} events in {}", format_ms(duration))}
            </p>
        </section>
        <section class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
                <h3 class="card-title text-sm">"Sites"</h3>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>"Site"</th>
                            <th>"Status"</th>
                            <th>"Router"</th>
                            <th>"Tool"</th>
                            <th class="text-right">"Routing"</th>
                            <th class="text-right">"Tokens"</th>
                            <th class="text-right">"Done at"</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sites.into_iter().map(|site| site_row(site, site_filter)).collect_view()}
                    </tbody>
                </table>
            </div>
        </section>
        {ranked.map(|(kept, excluded)| view! { <Products kept excluded /> })}
        <section class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
                <div class="flex items-center justify-between">
                    <h3 class="card-title text-sm">"Timeline"</h3>
                    <Show when=move || site_filter.get().is_some()>
                        <button class="btn btn-ghost btn-xs" on:click=move |_| site_filter.set(None)>
                            {move || format!("Only {} · show all", site_filter.get().unwrap_or_default())}
                        </button>
                    </Show>
                </div>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th class="text-right">"ms"</th>
                            <th>"Site"</th>
                            <th>"Event"</th>
                            <th>"Details"</th>
                        </tr>
                    </thead>
                    <tbody>
                        <For
                            each=rows
                            key=|(i, _)| *i
                            children=move |(i, record)| {
                                let detail = serde_json::to_string_pretty(&record).unwrap_or_default();
                                view! {
                                    <tr
                                        class="hover:bg-base-200 cursor-pointer"
                                        on:click=move |_| {
                                            open_row.update(|o| *o = if *o == Some(i) { None } else { Some(i) })
                                        }
                                    >
                                        <td class="text-right font-mono tnum">{record.t_ms}</td>
                                        <td class="font-mono">{record.site.clone().unwrap_or_default()}</td>
                                        <td>
                                            <span class=format!(
                                                "badge badge-sm whitespace-nowrap {}",
                                                summary::tone(&record.event),
                                            )>{record.event.name()}</span>
                                        </td>
                                        <td class="break-all">{summary::describe(&record.event)}</td>
                                    </tr>
                                    <Show when=move || open_row.get() == Some(i)>
                                        <tr>
                                            <td colspan="4">
                                                <pre class="text-xs font-mono max-h-96 overflow-auto bg-base-200 p-3 rounded">
                                                    {detail.clone()}
                                                </pre>
                                            </td>
                                        </tr>
                                    </Show>
                                }
                            }
                        />
                    </tbody>
                </table>
            </div>
        </section>
    }
}

fn site_row(site: SiteRun, filter: RwSignal<Option<String>>) -> impl IntoView {
    let name = site.site.clone();
    let status = site.status.clone().unwrap_or_else(|| "unfinished".into());
    let badge = format!("badge badge-sm {}", summary::status_tone(&status));
    view! {
        <tr class="hover:bg-base-200 cursor-pointer" on:click=move |_| filter.set(Some(name.clone()))>
            <td>
                <div class="font-mono">{site.site.clone()}</div>
                <div class="text-xs opacity-60 break-all">{site.url.clone()}</div>
            </td>
            <td>
                <span class=badge>{status}</span>
                <div class="text-xs opacity-80 break-all">{site.error.clone()}</div>
            </td>
            <td class="font-mono text-xs">{site.router.clone().unwrap_or_default()}</td>
            <td class="font-mono text-xs">{site.tool.clone().unwrap_or_else(|| "—".into())}</td>
            <td class="text-right tnum">{site.routing_ms.map(format_ms)}</td>
            <td class="text-right tnum">{site.tokens}</td>
            <td class="text-right tnum">{site.finished_ms.map(format_ms)}</td>
        </tr>
    }
}

#[component]
fn Products(
    kept: Vec<scout_types::RankedRef>,
    excluded: Vec<scout_types::Excluded>,
) -> impl IntoView {
    let more = kept.len().saturating_sub(TOP);
    let title = format!("Ranking · {} kept, {} excluded", kept.len(), excluded.len());
    view! {
        <section class="card bg-base-100 border border-base-300">
            <div class="card-body p-4">
                <h3 class="card-title text-sm">{title}</h3>
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th class="text-right">"#"</th>
                            <th class="text-right">"Score"</th>
                            <th>"Product"</th>
                            <th class="text-right">"Price"</th>
                            <th>"Source"</th>
                        </tr>
                    </thead>
                    <tbody>
                        {kept
                            .into_iter()
                            .take(TOP)
                            .enumerate()
                            .map(|(i, p)| {
                                let price = p
                                    .price
                                    .map(|m| format!("{:.2} {}", m.amount, m.currency))
                                    .unwrap_or_else(|| "—".into());
                                view! {
                                    <tr>
                                        <td class="text-right tnum">{i + 1}</td>
                                        <td class="text-right tnum">{format!("{:.2}", p.score)}</td>
                                        <td>{p.name}</td>
                                        <td class="text-right tnum whitespace-nowrap">{price}</td>
                                        <td class="font-mono text-xs">{format!("{} {}", p.source, p.id)}</td>
                                    </tr>
                                }
                            })
                            .collect_view()}
                    </tbody>
                </table>
                {(more > 0).then(|| view! { <p class="text-xs opacity-60">{format!("… {more} more")}</p> })}
                {(!excluded.is_empty())
                    .then(|| {
                        view! {
                            <details class="text-sm">
                                <summary class="cursor-pointer opacity-80">"Excluded"</summary>
                                <ul class="mt-2 space-y-1">
                                    {excluded
                                        .into_iter()
                                        .map(|e| {
                                            view! {
                                                <li>
                                                    <span class="font-mono text-xs">
                                                        {format!("{} {}", e.source, e.id)}
                                                    </span>
                                                    " — "
                                                    {e.reason}
                                                </li>
                                            }
                                        })
                                        .collect_view()}
                                </ul>
                            </details>
                        }
                    })}
            </div>
        </section>
    }
}

fn format_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else {
        format!("{:.1} s", ms as f64 / 1000.0)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    }
}

fn format_modified(secs: u64) -> String {
    if secs == 0 {
        return "sample".into();
    }
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(secs as f64 * 1000.0));
    // Swedish locale: ISO-like "2026-09-26 10:09:30".
    String::from(date.to_locale_string("sv-SE", &wasm_bindgen::JsValue::UNDEFINED))
}
