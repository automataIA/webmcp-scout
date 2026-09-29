# Threat model

Scope: the MVP runtime (`crates/scout`, `scout` CLI). It drives a local Chrome over CDP, reads the WebMCP tools of configured sites, asks models to pick a tool and its arguments, and executes **read-only, allowlisted** tools. The desktop trace viewer (M7) has its own section below.

## Assets

- The user's machine and accounts: Chrome must never act with the user's identity.
- API keys (`TYPESAFE_API_KEY`, Groq), held in `.env` and in the Rust process only.
- The integrity of the decision: which site is queried, which tool runs, with which arguments.
- The results shown to the user and the traces written to `traces/`.

## Trust boundaries

| Source | Trusted | Why |
|---|---|---|
| `sites.toml` (sites, origins, `(tool, schema_hash, class, adapter)`) | yes | written and reviewed by the user |
| The user's request | yes, as intent | it still passes through the same policy |
| Page content: DOM, tool names, titles, descriptions, input schemas, tool results | **no** | controlled by the site, or by whoever compromised it |
| Model output (Groq, Jev) | **no** | the models read untrusted text and can be steered by it |
| Chrome and CDP | yes | local binary, launched by us with a throwaway profile |

Every model reads tool descriptions and schemas, so a "data field" is not a safe channel: injected text reaches Jev and the LLM alike. The defence is that nothing a model or a page says can widen what the runtime is allowed to do.

## Threats and mitigations

| # | Threat | Mitigation | Verified by |
|---|---|---|---|
| T1 | Prompt injection in tool metadata makes the model pick a dangerous tool (`checkout`, `purchase_cart`) | Only allowlisted tools are offered to any model; a tool name the router returns outside the offered set is invalid and never executed; the MVP allowlist admits `class = "read_only"` only | `tests/adversarial.rs` (`a_model_fooled_by_injected_metadata_cannot_run_checkout`), `tests/ask.rs` |
| T2 | Prompt injection in tool results steers a follow-up action | A request is single-shot: one routing decision, one execution; results go to the adapter and the ranking, never back to a model | `tests/adversarial.rs` (`injected_results_are_data_and_trigger_no_further_call`) |
| T3 | The page changes a tool's schema (new parameter, different semantics) after it was allowlisted | The allowlist pins `schema_hash`; a different hash at discovery excludes the tool before routing | `tests/ask.rs` (`changed_schema_hash_blocks_the_tool_before_routing`) |
| T4 | The page swaps or drops the tool between routing and execution | Re-discovery right before execution: the tool must still be registered, on the site origin, with the allowlisted hash | `tests/adversarial.rs` (drift tests) |
| T5 | Arguments outside the schema (negative prices, unknown keys, injected strings in closed fields) | Router output is validated against the site schema subset before execution; invalid arguments execute nothing | `tests/ask.rs` (`no_tool_and_invalid_arguments_execute_nothing`), router unit tests |
| T6 | The page redirects to another origin | Discovery and re-discovery must report the site origin; navigation during a call is recorded in the trace | `run.rs`, `tests/fixtures.rs` |
| T7 | Tool output or descriptions change policy, allowlists or site selection | Sites and capabilities come only from `sites.toml` and the CLI; no code path writes policy from page or model data | code review |
| T8 | Using the user's logged-in sessions | Chrome runs with a new temporary profile per run, removed on exit; never a personal profile | `browser` module, `apps/scout-cli/tests/interrupt.rs` |
| T9 | Chrome left running after Ctrl-C, SIGTERM or a crash of a command | Signals are caught before launch; the command is dropped and Chrome shut down (killed after a grace period), then the profile is removed once no process uses it | `apps/scout-cli/tests/interrupt.rs` |
| T10 | Key leakage | Keys only in the Rust process, redacted in `Debug`, never traced; `.env` is git-ignored | code review |
| T11 | A slow or hostile site stalls the run | Per-call timeout and per-site budget; a timed-out site is reported and the others keep their results | `tests/multi_site.rs` |
| T12 | Oversized tool descriptions or outputs | Descriptions are truncated before they reach Jev; trace previews are capped at 2000 characters | `router/hybrid.rs`, `trace.rs` |


## Desktop viewer

The viewer (`apps/desktop`) reads traces and starts `scout ask`; it never drives Chrome. Traces are untrusted: they hold tool descriptions, outputs and product names written by the pages.

| # | Threat | Mitigation |
|---|---|---|
| D1 | Markup or script in a trace (a product name, a tool output) runs in the webview | Leptos renders every trace string as a text node, never as HTML; product URLs are never turned into links; the CSP allows only the app's own script and wasm (`script-src 'self' 'wasm-unsafe-eval'`), no inline code, no remote connections besides IPC; `freezePrototype` is on. Checked with a trace carrying `<img onerror>` and `<script>`: no element is created |
| D2 | A compromised webview reads arbitrary files through IPC | One capability grants only `list_traces`, `read_trace`, `start_run` and event listening; `read_trace` accepts a plain `.jsonl` file name (no separators, no leading dot) inside `traces/` |
| D3 | A request typed in the viewer becomes a `scout` option | The child is spawned without a shell, with the request after `--`, stdin closed |
| D4 | Runs pile up or outlive the window | One run at a time; on exit the viewer sends SIGTERM to its own child, and `scout` shuts Chrome down and removes the profile (T9) |

Residual: the viewer runs `target/{release,debug}/scout` of the project directory, or `SCOUT_BIN`; whoever controls those paths controls what runs.

## Residual risks

- **The page reports its own schema.** `schema_hash` covers what `document.modelContext.getTools()` returns. A hostile page could report one schema and implement another behind `execute`; the hash proves only that the declared interface did not change. Read-only classification is a promise by the site, checked by the user when writing `sites.toml`, not by the runtime.
- **Time of check to time of use.** The page can still swap a tool in the few milliseconds between re-discovery and `executeTool`. Re-discovery narrows the window; it cannot close it from outside the page.
- **Injected text reaches third-party models.** Tool metadata and the request are sent to Groq and TypeSafe. Injection can degrade the routing (wrong allowlisted tool, wrong arguments within the schema), which is a quality risk, not a privilege escalation.
- **Results shown to the user carry the site's text**, injection included. They are displayed as data; a user or a later agent consuming them must treat them as untrusted.
- **Chrome itself.** A Chrome exploit in a hostile page is outside what this runtime can mitigate beyond headless mode, a temporary profile and keeping Chrome up to date.
- **Non-Linux hosts.** Waiting for Chrome helpers before removing the profile reads `/proc`; elsewhere the profile is removed right after Chrome exits.
