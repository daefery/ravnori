//! Runs that keep no transcript (`codex exec --ephemeral`, `pi --no-session`) are scripted
//! automation: they get the user's pinned facts and no memories, and no offers are
//! recorded for them. Sessions that keep a transcript, including a new Claude Code
//! session whose transcript file does not exist yet, are served exactly as before.
//! Runs the real binary, as the agents call it.

use rusqlite::params;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const PROJECT: &str = "/work/proj";

fn rvn(home: &Path, args: &[&str], stdin: Option<&str>) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvn"))
        .env("HOME", home)
        .env("RAVNORI_HOME", home.join(".ravnori"))
        .env("RAVNORI_UI_PORT", "0")
        .env_remove("CLAUDE_CONFIG_DIR")
        .arg("--db")
        .arg(home.join("ravnori.db"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    if let Some(s) = stdin {
        // A hook may exit before reading everything; a broken pipe is not a failure.
        let _ = input.write_all(s.as_bytes());
    }
    drop(input);
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "rvn {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A home with one pinned fact and one observation in PROJECT, whose cwd is not a git
/// repository, so the project is the directory itself.
fn setup(name: &str) -> (ravnori::TempDir, std::path::PathBuf) {
    let base = ravnori::TempDir::new(&format!("ephemeral-{name}"));
    let home = base.join("home");
    std::fs::create_dir_all(home.join(".ravnori")).unwrap();
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    ravnori::forget::remember(&c, "Never add AI trailers to commits", None).unwrap();
    c.execute(
        "INSERT INTO memories(id, session_id, project, kind, type, title, origin, origin_id, created_at)
         VALUES (500, 'claude:earlier', ?1, 'observation', 'decision', 'Retry loop must stay bounded', 'ravnori', 500, ?2)",
        params![PROJECT, ravnori::db::now_ms() - 60_000],
    )
    .unwrap();
    (base, home)
}

fn offers(home: &Path, session: &str) -> Vec<(i64, String)> {
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    let mut st = c
        .prepare("SELECT memory_id, source FROM offers WHERE session_id = ?1 ORDER BY 1, 2")
        .unwrap();
    st.query_map([session], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn context_of(hook_out: &str) -> String {
    let v: serde_json::Value =
        serde_json::from_str(hook_out.trim()).unwrap_or_else(|_| panic!("not JSON: {hook_out}"));
    v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn an_ephemeral_codex_run_gets_pins_only_and_records_nothing() {
    let (_base, home) = setup("codex");
    let payload = |event: &str| {
        serde_json::json!({
            "session_id": "eph", "transcript_path": null, "cwd": PROJECT,
            "hook_event_name": event, "source": "startup",
            "prompt": "fix the retry loop"
        })
        .to_string()
    };
    let start = rvn(
        &home,
        &["hook", "codex", "session-start"],
        Some(&payload("SessionStart")),
    );
    let ctx = context_of(&start);
    assert!(
        ctx.contains("Never add AI trailers"),
        "pins still apply: {ctx}"
    );
    assert!(!ctx.contains("#500"), "no memories: {ctx}");
    assert!(!ctx.contains("Recent sessions"), "{ctx}");
    let prompt = rvn(
        &home,
        &["hook", "codex", "prompt"],
        Some(&payload("UserPromptSubmit")),
    );
    assert_eq!(prompt.trim(), "", "no prompt recall in an ephemeral run");
    assert!(offers(&home, "codex:eph").is_empty());
    // No watermark either: nothing was injected.
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    let n: i64 = c
        .query_row(
            "SELECT count(*) FROM injections WHERE session_id = 'codex:eph'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0);
}

#[test]
fn an_ephemeral_run_with_nothing_pinned_gets_nothing() {
    let (_base, home) = setup("nopins");
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    c.execute("DELETE FROM memories WHERE kind = 'pinned'", [])
        .unwrap();
    drop(c);
    let out = rvn(
        &home,
        &["hook", "codex", "session-start"],
        Some(
            &serde_json::json!({"session_id": "eph", "transcript_path": null, "cwd": PROJECT})
                .to_string(),
        ),
    );
    assert_eq!(out.trim(), "");
}

/// Claude Code sends the path of a transcript it has not written yet on SessionStart and
/// the first prompt, for every new session: that is not an ephemeral run.
#[test]
fn a_session_whose_transcript_is_not_written_yet_is_served_as_before() {
    let (_base, home) = setup("claude");
    let not_yet = home.join("not-yet-written.jsonl");
    let payload = serde_json::json!({
        "session_id": "new", "transcript_path": not_yet.to_string_lossy(), "cwd": PROJECT,
        "hook_event_name": "SessionStart", "source": "startup"
    })
    .to_string();
    let ctx = context_of(&rvn(
        &home,
        &["hook", "claude", "session-start"],
        Some(&payload),
    ));
    assert!(
        ctx.contains("Never add AI trailers") && ctx.contains("#500"),
        "{ctx}"
    );
    assert_eq!(offers(&home, "claude:new"), [(500, "start".to_string())]);
    // A payload without the field at all (older agents) is served as before too.
    let bare = serde_json::json!({"session_id": "bare", "cwd": PROJECT}).to_string();
    let ctx = context_of(&rvn(
        &home,
        &["hook", "claude", "session-start"],
        Some(&bare),
    ));
    assert!(ctx.contains("#500"), "{ctx}");
}

/// pi's extension: `rvn context --pins-only` for `pi --no-session`, and the start
/// context's memories recorded as offered when `rvn delta` delivers it.
#[test]
fn pi_pins_only_context_and_start_offers() {
    let (_base, home) = setup("pi");
    let pins = rvn(&home, &["context", "--cwd", PROJECT, "--pins-only"], None);
    assert!(pins.contains("Never add AI trailers"), "{pins}");
    assert!(
        !pins.contains("#500") && !pins.contains("Recent sessions"),
        "{pins}"
    );

    let full = rvn(
        &home,
        &["context", "--cwd", PROJECT, "--session", "pi:p1"],
        None,
    );
    assert!(full.contains("#500"), "{full}");
    // Printing the context records nothing: pi may never deliver it.
    assert!(offers(&home, "pi:p1").is_empty());
    rvn(
        &home,
        &[
            "delta",
            "--session",
            "pi:p1",
            "--cwd",
            PROJECT,
            "--start-ids",
            "500",
        ],
        None,
    );
    assert_eq!(offers(&home, "pi:p1"), [(500, "start".to_string())]);
    // The first delta still sets the watermark, as before.
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    let n: i64 = c
        .query_row(
            "SELECT count(*) FROM injections WHERE session_id = 'pi:p1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1);
}
