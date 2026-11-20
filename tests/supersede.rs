//! A memory a newer one made outdated (a link in `supersedes`) is not what automatic
//! recall shows: prompt recall gives its place to the newer memory, the session start
//! leaves it out. Search and get_observations still show it, marked, and
//! `recall.hide_replaced: false` shows both as before. Runs the real binary.

use rusqlite::params;
use std::path::Path;
use std::process::Command;

const PROJECT: &str = "/work/proj";

fn rvn(home: &Path, config: Option<&str>, args: &[&str]) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rvn"));
    cmd.env("HOME", home)
        .env("RAVNORI_HOME", home.join(".ravnori"))
        .env("RAVNORI_UI_PORT", "0")
        .env_remove("CLAUDE_CONFIG_DIR")
        .arg("--db")
        .arg(home.join("ravnori.db"))
        .args(args);
    let cfg = home.join("config.json");
    std::fs::write(
        &cfg,
        config.unwrap_or(r#"{"semantic": {"enabled": false}}"#),
    )
    .unwrap();
    cmd.env("RAVNORI_CONFIG", &cfg);
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "rvn {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Memory 500 (two days old) says the retry queue is blocked; 501 (yesterday) replaced it:
/// the retry queue is unblocked. 502 is unrelated and stays.
fn setup(name: &str) -> (ravnori::TempDir, std::path::PathBuf) {
    let base = ravnori::TempDir::new(&format!("supersede-{name}"));
    let home = base.join("home");
    std::fs::create_dir_all(home.join(".ravnori")).unwrap();
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    let now = ravnori::db::now_ms();
    for (id, title, ago_h) in [
        (500, "Retry queue drain blocked by stuck lease worker", 48),
        (
            501,
            "Retry queue drain unblocked after lease worker fix",
            24,
        ),
        (502, "Billing export writes monthly CSV files", 30),
    ] {
        c.execute(
            "INSERT INTO memories(id, session_id, project, kind, type, title, narrative, origin, origin_id, created_at)
             VALUES (?1, 'claude:earlier', ?2, 'observation', 'discovery', ?3, ?3, 'ravnori', ?1, ?4)",
            params![id, PROJECT, title, now - ago_h * 3_600_000],
        )
        .unwrap();
    }
    // Other work in other projects, so the prompt's words are rare, as in a real record
    // (prompt recall needs a term found in under 10% of memories).
    for i in 0..40 {
        c.execute(
            "INSERT INTO memories(id, session_id, project, kind, type, title, origin, origin_id, created_at)
             VALUES (?1, 'claude:other', '/work/other', 'observation', 'change', ?2, 'ravnori', ?1, ?3)",
            params![600 + i, format!("Dashboard widget {i} renders totals"), now - 72 * 3_600_000],
        )
        .unwrap();
    }
    c.execute(
        "INSERT INTO supersedes(old_id, new_id, reason, model, at) VALUES (500, 501, 'the worker was fixed', 'm', 0)",
        [],
    )
    .unwrap();
    (base, home)
}

fn recall(home: &Path, config: Option<&str>, session: &str, prompt: &str) -> String {
    rvn(
        home,
        config,
        &[
            "recall-probe",
            "--session",
            session,
            "--project",
            PROJECT,
            "--prompt",
            prompt,
        ],
    )
}

const PROMPT: &str = "why is the retry queue drain blocked by the stuck lease worker";

#[test]
fn prompt_recall_shows_the_newer_memory_in_place_of_the_one_it_replaced() {
    let (_base, home) = setup("prompt");
    let out = recall(&home, None, "claude:now", PROMPT);
    assert!(out.contains("#501"), "the newer memory: {out}");
    assert!(!out.contains("#500"), "not the replaced one: {out}");
    assert_eq!(out.matches("#501").count(), 1, "once: {out}");
    // Asked again in the same session: the newer one was already shown, nothing repeats.
    let again = recall(&home, None, "claude:now", PROMPT);
    assert!(
        !again.contains("#500") && !again.contains("#501"),
        "{again}"
    );
}

#[test]
fn the_session_start_lists_the_newer_memory_only() {
    let (_base, home) = setup("start");
    let ctx = rvn(&home, None, &["context", "--project", PROJECT]);
    assert!(ctx.contains("#501") && ctx.contains("#502"), "{ctx}");
    assert!(!ctx.contains("#500"), "{ctx}");
}

#[test]
fn search_and_get_observations_still_show_the_replaced_memory_marked() {
    let (_base, home) = setup("search");
    let found = rvn(
        &home,
        None,
        &[
            "tool",
            "search",
            &format!(r#"{{"query": "retry queue lease worker", "project": "{PROJECT}"}}"#),
        ],
    );
    assert!(found.contains("#500"), "{found}");
    assert!(found.contains("replaced by #501"), "{found}");
    let full = rvn(
        &home,
        None,
        &["tool", "get_observations", r#"{"ids": [500, 501]}"#],
    );
    assert!(
        full.contains("Replaced by #501") && full.contains("the worker was fixed"),
        "{full}"
    );
    assert!(full.contains("Replaces #500"), "{full}");
}

#[test]
fn hide_replaced_off_shows_both_as_before() {
    let (_base, home) = setup("off");
    let off = r#"{"semantic": {"enabled": false}, "recall": {"hide_replaced": false}}"#;
    let out = recall(&home, Some(off), "claude:off", PROMPT);
    assert!(out.contains("#500"), "{out}");
    let ctx = rvn(&home, Some(off), &["context", "--project", PROJECT]);
    assert!(ctx.contains("#500") && ctx.contains("#501"), "{ctx}");
}

#[test]
fn a_replayed_prompt_does_not_see_a_replacement_written_after_it() {
    let (_base, home) = setup("replay");
    let c = ravnori::db::open(&home.join("ravnori.db")).unwrap();
    let before = ravnori::db::now_ms() - 36 * 3_600_000; // between 500 and 501
    let scope = ravnori::recall::Scope {
        session: None,
        offered_to: None,
        before: Some(before),
    };
    let rows = vec![(500, "discovery".to_string(), "old".to_string(), 0)];
    let kept = ravnori::recall::with_replacements(&c, &scope, rows.clone()).unwrap();
    assert_eq!(kept[0].0, 500, "501 did not exist yet");
    let now = ravnori::recall::Scope::default();
    assert_eq!(
        ravnori::recall::with_replacements(&c, &now, rows).unwrap()[0].0,
        501
    );
}
