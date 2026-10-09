//! Local Claude Code CLI integration.
//!
//! Detects the `claude` binary and runs it in headless streaming mode against the
//! knowledge vault (`cwd = vault`, writes scoped via `--add-dir`, `acceptEdits`).
//! stdout is JSON-lines (`--output-format stream-json`); we parse each line and
//! emit Tauri events (`claude-delta`, `claude-tool`, `claude-result`,
//! `claude-error`) so the UI can stream the assistant's work. The returned
//! `session_id` is persisted so chat follow-ups can `--resume`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter};

use crate::storage;

#[derive(Debug, Clone, Serialize)]
pub struct ClaudeInfo {
    pub path: String,
    pub version: String,
}

/// Candidate locations — GUI apps on macOS don't inherit the shell PATH, so we
/// probe explicit paths in addition to `claude` on PATH.
fn candidates(explicit: Option<&str>) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Some(e) = explicit {
        if !e.is_empty() {
            v.push(PathBuf::from(e));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        v.push(PathBuf::from(format!("{home}/.local/bin/claude")));
        v.push(PathBuf::from(format!("{home}/.claude/local/claude")));
    }
    v.push(PathBuf::from("claude"));
    v.push(PathBuf::from("/opt/homebrew/bin/claude"));
    v.push(PathBuf::from("/usr/local/bin/claude"));
    v
}

/// Find a working `claude` binary and its version.
pub fn detect(explicit: Option<&str>) -> Result<ClaudeInfo> {
    for cand in candidates(explicit) {
        if let Ok(out) = Command::new(&cand).arg("--version").output() {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
                return Ok(ClaudeInfo {
                    path: cand.to_string_lossy().to_string(),
                    version,
                });
            }
        }
    }
    Err(anyhow!(
        "Claude Code CLI not found. Install it and run `claude` once to log in."
    ))
}

pub struct RunOpts {
    pub claude_path: String,
    pub vault: PathBuf,
    pub db_path: PathBuf,
    /// Our app session id — used to tag events and persist chat history.
    pub app_session: String,
    pub prompt: String,
    pub model: String,
    pub resume_sid: Option<String>,
    pub max_budget: Option<f64>,
    /// Extra tool permissions on top of the built-in file tools (settings + one-shot grants).
    pub extra_tools: Vec<String>,
    /// Extra system prompt (the vault's `agent.md`), appended to Claude's default.
    pub append_system: Option<String>,
    /// Routing tag for UI events: "chat" (assistant panel) or "story" (story tab).
    pub kind: String,
    /// Persist the claude session id + reply into chat history (chat only).
    pub persist: bool,
}

#[derive(Clone, Serialize)]
struct DeltaEvent {
    session: String,
    kind: String,
    text: String,
}
#[derive(Clone, Serialize)]
struct ToolEvent {
    session: String,
    kind: String,
    name: String,
    file: Option<String>,
}
/// A tool call Claude Code refused because it wasn't permitted. The UI offers
/// to grant `rule` (an `--allowedTools` entry) once or permanently.
#[derive(Clone, Serialize, PartialEq)]
pub struct Denial {
    pub tool: String,
    pub rule: String,
    pub detail: String,
}
#[derive(Clone, Serialize)]
struct ResultEvent {
    session: String,
    kind: String,
    text: String,
    cost_usd: Option<f64>,
    denials: Vec<Denial>,
}

/// Turn a `permission_denials` entry into a grantable rule. Bash is narrowed to
/// the command's first word (`Bash(git *)`) rather than all of Bash.
fn denial_from(v: &Value) -> Option<Denial> {
    let tool = v.get("tool_name")?.as_str()?.to_string();
    let input = v.get("tool_input");
    let get = |k: &str| input.and_then(|i| i.get(k)).and_then(|x| x.as_str());
    let (rule, detail) = match tool.as_str() {
        "Bash" => {
            let cmd = get("command").unwrap_or("");
            let first = cmd.split_whitespace().next().unwrap_or("");
            let rule = if first.is_empty() { "Bash".to_string() } else { format!("Bash({first} *)") };
            (rule, cmd.to_string())
        }
        "WebSearch" => (tool.clone(), get("query").unwrap_or("").to_string()),
        "WebFetch" => (tool.clone(), get("url").unwrap_or("").to_string()),
        _ => (
            tool.clone(),
            get("file_path").or_else(|| get("path")).unwrap_or("").to_string(),
        ),
    };
    Some(Denial { tool, rule, detail })
}

const BASE_TOOLS: &[&str] = &["Read", "Write", "Edit", "Glob", "Grep"];

/// Comma-joined `--allowedTools` value: built-in file tools + extras (deduped).
fn allowed_tools_arg(extra: &[String]) -> String {
    let mut v: Vec<String> = BASE_TOOLS.iter().map(|s| s.to_string()).collect();
    for e in extra {
        let e = e.trim();
        if !e.is_empty() && !v.iter().any(|x| x == e) {
            v.push(e.to_string());
        }
    }
    v.join(",")
}

#[derive(Clone, Serialize)]
pub struct ModelInfo {
    pub value: String,
    pub display_name: String,
    pub description: String,
    pub resolved: String,
}

/// Ask the installed Claude Code for the models it actually offers, via the
/// stream-json control protocol (`initialize` → `models`). Same data `/model` shows.
pub fn list_models(claude_path: &str) -> Result<Vec<ModelInfo>> {
    use std::io::Write;
    let mut cmd = Command::new(claude_path);
    cmd.arg("-p")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    scrub_env(&mut cmd);
    let mut child = cmd.spawn().context("launch claude for model list")?;
    let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    stdin.write_all(
        b"{\"type\":\"control_request\",\"request_id\":\"vexa-models\",\"request\":{\"subtype\":\"initialize\"}}\n",
    )?;
    stdin.flush()?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;

    // Read on a thread so a hung CLI can't block us past the timeout.
    let (tx, rx) = std::sync::mpsc::channel::<Option<Value>>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                if v.get("type").and_then(|t| t.as_str()) == Some("control_response") {
                    let _ = tx.send(Some(v));
                    return;
                }
            }
        }
        let _ = tx.send(None);
    });
    let resp = rx.recv_timeout(std::time::Duration::from_secs(25));
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    let v = resp
        .map_err(|_| anyhow!("timed out waiting for Claude Code"))?
        .ok_or_else(|| anyhow!("Claude Code exited without answering"))?;
    let models = v
        .pointer("/response/response/models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow!("Claude Code returned no model list"))?;
    let s = |m: &Value, k: &str| m.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let out: Vec<ModelInfo> = models
        .iter()
        .filter_map(|m| {
            let value = s(m, "value");
            if value.is_empty() {
                return None;
            }
            Some(ModelInfo {
                display_name: s(m, "displayName"),
                description: s(m, "description"),
                resolved: s(m, "resolvedModel"),
                value,
            })
        })
        .collect();
    if out.is_empty() {
        return Err(anyhow!("Claude Code returned an empty model list"));
    }
    Ok(out)
}
#[derive(Clone, Serialize)]
struct ErrEvent {
    session: String,
    kind: String,
    message: String,
}

/// Spawn a Claude Code run on a background thread. Events stream to the UI; on
/// completion the claude session id + assistant reply are persisted.
pub fn spawn_run(app: AppHandle, opts: RunOpts) {
    std::thread::Builder::new()
        .name("vexa-claude".into())
        .spawn(move || {
            if let Err(e) = run(&app, &opts) {
                let _ = app.emit(
                    "claude-error",
                    ErrEvent {
                        session: opts.app_session.clone(),
                        kind: opts.kind.clone(),
                        message: format!("{e:#}"),
                    },
                );
            }
        })
        .ok();
}

/// Strip env that would hijack the spawned `claude`'s auth/endpoint so it always
/// uses the user's own Claude Code login (keychain/OAuth). Critical when the app
/// is launched from a shell that itself runs inside Claude Code.
fn scrub_env(cmd: &mut Command) {
    for (k, _) in std::env::vars() {
        let up = k.to_uppercase();
        if up.starts_with("CLAUDE")
            || up == "ANTHROPIC_BASE_URL"
            || up == "ANTHROPIC_API_KEY"
            || up == "ANTHROPIC_AUTH_TOKEN"
            || up == "ANTHROPIC_MODEL"
            || up == "ANTHROPIC_SMALL_FAST_MODEL"
        {
            cmd.env_remove(k);
        }
    }
}

/// One-shot Claude call (text output) that can Read images — used for speaker
/// naming from frames. Returns stdout.
pub fn run_sync(
    claude_path: &str,
    cwd: &Path,
    add_dir: &Path,
    model: &str,
    prompt: &str,
) -> Result<String> {
    let mut cmd = Command::new(claude_path);
    cmd.current_dir(cwd)
        .arg("-p")
        .arg("--add-dir")
        .arg(add_dir)
        .arg("--allowedTools")
        .arg("Read")
        .arg("--model")
        .arg(model)
        .arg(prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    scrub_env(&mut cmd);
    let out = cmd.output().context("run claude (vision)")?;
    if !out.status.success() {
        return Err(anyhow!(
            "claude failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Minimal one-shot text completion: `claude -p --model <m> <prompt>` with no
/// tools or dir access. Used for transcript polishing (text in → markdown out).
pub fn run_prompt(claude_path: &str, model: &str, prompt: &str) -> Result<String> {
    let mut cmd = Command::new(claude_path);
    cmd.arg("-p")
        .arg("--model")
        .arg(model)
        .arg(prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    scrub_env(&mut cmd);
    let out = cmd.output().context("run claude (polish)")?;
    if !out.status.success() {
        return Err(anyhow!(
            "claude failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run(app: &AppHandle, opts: &RunOpts) -> Result<()> {
    let mut cmd = Command::new(&opts.claude_path);
    cmd.current_dir(&opts.vault)
        .arg("-p")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--add-dir")
        .arg(&opts.vault)
        .arg("--permission-mode")
        .arg("acceptEdits")
        .arg("--allowedTools")
        .arg(allowed_tools_arg(&opts.extra_tools));
    // "default" = whatever Claude Code picks for this account; omit --model.
    if !opts.model.is_empty() && opts.model != "default" {
        cmd.arg("--model").arg(&opts.model);
    }
    if let Some(sys) = &opts.append_system {
        if !sys.trim().is_empty() {
            cmd.arg("--append-system-prompt").arg(sys);
        }
    }
    if let Some(sid) = &opts.resume_sid {
        cmd.arg("--resume").arg(sid);
    }
    if let Some(b) = opts.max_budget {
        cmd.arg("--max-budget-usd").arg(format!("{b}"));
    }
    cmd.arg(&opts.prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    scrub_env(&mut cmd);

    let mut child = cmd.spawn().with_context(|| {
        format!("failed to launch claude at {}", opts.claude_path)
    })?;

    // Drain stderr on a thread so it can't deadlock the pipe.
    let stderr = child.stderr.take();
    let err_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut e) = stderr {
            use std::io::Read;
            let _ = e.read_to_string(&mut buf);
        }
        buf
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no stdout from claude"))?;
    let reader = BufReader::new(stdout);

    let mut claude_sid: Option<String> = None;
    let mut final_text = String::new();
    let mut cost: Option<f64> = None;
    let mut denials: Vec<Denial> = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue, // ignore non-JSON noise
        };
        if let Some(sid) = v.get("session_id").and_then(|s| s.as_str()) {
            claude_sid = Some(sid.to_string());
        }
        match v.get("type").and_then(|t| t.as_str()) {
            Some("assistant") => {
                if let Some(content) = v
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_array())
                {
                    for item in content {
                        match item.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                    let _ = app.emit(
                                        "claude-delta",
                                        DeltaEvent {
                                            session: opts.app_session.clone(),
                                            kind: opts.kind.clone(),
                                            text: t.to_string(),
                                        },
                                    );
                                }
                            }
                            Some("tool_use") => {
                                let name = item
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("tool")
                                    .to_string();
                                let file = item
                                    .get("input")
                                    .and_then(|i| {
                                        i.get("file_path").or_else(|| i.get("path"))
                                    })
                                    .and_then(|p| p.as_str())
                                    .map(|s| s.to_string());
                                let _ = app.emit(
                                    "claude-tool",
                                    ToolEvent {
                                        session: opts.app_session.clone(),
                                        kind: opts.kind.clone(),
                                        name,
                                        file,
                                    },
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some("result") => {
                final_text = v
                    .get("result")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string();
                cost = v.get("total_cost_usd").and_then(|c| c.as_f64());
                if let Some(arr) = v.get("permission_denials").and_then(|d| d.as_array()) {
                    for d in arr.iter().filter_map(denial_from) {
                        if !denials.iter().any(|x| x.rule == d.rule) {
                            denials.push(d);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let status = child.wait().ok();
    let stderr_text = err_handle.join().unwrap_or_default();

    // Persist claude session id (for resume) + the assistant reply (chat only).
    if opts.persist {
        if let Ok(conn) = storage::open(&opts.db_path) {
            if let Some(sid) = &claude_sid {
                let _ = storage::set_claude_session(&conn, &opts.app_session, sid);
            }
            if !final_text.is_empty() {
                let now = chrono::Utc::now().to_rfc3339();
                let _ =
                    storage::insert_chat(&conn, &opts.app_session, "assistant", &final_text, &now);
            }
        }
    }

    let ok = status.map(|s| s.success()).unwrap_or(false);
    if !ok && final_text.is_empty() {
        return Err(anyhow!(
            "claude exited without a result{}",
            if stderr_text.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", stderr_text.trim())
            }
        ));
    }

    let _ = app.emit(
        "claude-result",
        ResultEvent {
            session: opts.app_session.clone(),
            kind: opts.kind.clone(),
            text: final_text,
            cost_usd: cost,
            denials,
        },
    );
    Ok(())
}
