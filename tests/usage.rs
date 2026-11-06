//! `rvn usage` adds up the tokens in Claude Code, Codex and pi transcripts per week,
//! agent, route and model family, counting each reply once however often its transcript
//! repeats it. Runs the real binary against a scratch home.

use std::path::Path;
use std::process::Command;

fn write(path: &Path, lines: &[String]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, lines.join("\n") + "\n").unwrap();
}

fn now_iso(minus_min: i64) -> String {
    let ms = ravnori::db::now_ms() - minus_min * 60_000;
    let days = ms.div_euclid(86_400_000);
    let (y, m, d) = ravnori::text::civil_from_days(days);
    let s = ms.rem_euclid(86_400_000) / 1000;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.000Z",
        s / 3600,
        s / 60 % 60,
        s % 60
    )
}

fn usage(home: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_rvn"))
        .env("HOME", home)
        .env("RAVNORI_HOME", home.join(".ravnori"))
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("RAVNORI_CLAUDE_DIRS")
        .arg("--db")
        .arg(home.join("ravnori.db"))
        // Two weeks, so "a few minutes ago" is inside even just after Monday midnight.
        .args(["usage", "--weeks", "2"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The table row whose agent and route columns are `agent` and `route`.
/// Its fields from the agent column on.
fn row<'a>(out: &'a str, agent: &str, route: &str) -> Vec<&'a str> {
    out.lines()
        .find_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let i = f.windows(2).position(|w| w[0] == agent && w[1] == route)?;
            Some(f[i..].to_vec())
        })
        .unwrap_or_else(|| panic!("no {agent} {route} row in:\n{out}"))
}

#[test]
fn tokens_are_summed_per_agent_and_route_and_each_reply_counted_once() {
    let base = ravnori::TempDir::new("usage");
    let home = base.join("home");
    let t = now_iso(5);
    let t2 = now_iso(4);
    let ms = ravnori::db::now_ms() - 5 * 60_000;

    // pi: two replies through a proxy account, one copied into a forked session, and one
    // direct reply without a prefix.
    let pi = |rid: &str, model: &str, input: u64| {
        format!(
            r#"{{"type":"message","id":"a","timestamp":"{t}","message":{{"role":"assistant","content":[],"provider":"cliproxyapi","model":"{model}","responseId":"{rid}","timestamp":{ms},"usage":{{"input":{input},"output":10,"cacheRead":1000,"cacheWrite":0,"totalTokens":0}}}}}}"#
        )
    };
    write(
        &home.join(".pi/agent/sessions/--w--/a.jsonl"),
        &[
            r#"{"type":"session","id":"s1","cwd":"/w"}"#.to_string(),
            pi("r1", "developer/claude-opus-5-5", 100),
            pi("r2", "developer/claude-opus-5-5", 200),
            pi("r3", "gpt-5.6-sol", 7),
            "{\"type\":\"message\",\"message\":{\"role\":\"assistant\",\"us".into(),
        ],
    );
    write(
        &home.join(".pi/agent/sessions/--w--/b-fork.jsonl"),
        &[pi("r1", "developer/claude-opus-5-5", 100)],
    );

    // Claude Code: one reply streamed as two lines with the same message id.
    let cc = |uuid: &str| {
        format!(
            r#"{{"type":"assistant","uuid":"{uuid}","sessionId":"c1","timestamp":"{t}","message":{{"id":"msg_1","model":"claude-opus-5-5","usage":{{"input_tokens":3,"output_tokens":50,"cache_read_input_tokens":400,"cache_creation_input_tokens":60}}}}}}"#
        )
    };
    write(
        &home.join(".claude/projects/-w/c1.jsonl"),
        &[cc("u1"), cc("u2")],
    );
    std::fs::write(
        home.join(".claude.json"),
        r#"{"oauthAccount":{"emailAddress":"dev@example.org"}}"#,
    )
    .unwrap();

    // Codex: input includes cached input; a repeated total adds nothing.
    let tc = |at: &str, total: u64, input: u64, cached: u64| {
        format!(
            r#"{{"timestamp":"{at}","type":"event_msg","payload":{{"type":"token_count","info":{{"total_token_usage":{{"total_tokens":{total}}},"last_token_usage":{{"input_tokens":{input},"cached_input_tokens":{cached},"output_tokens":5}}}}}}}}"#
        )
    };
    write(
        &home.join(".codex/sessions/2026/11/05/x.jsonl"),
        &[
            format!(
                r#"{{"timestamp":"{t}","type":"session_meta","payload":{{"model_provider":"openai"}}}}"#
            ),
            format!(
                r#"{{"timestamp":"{t}","type":"turn_context","payload":{{"model":"gpt-6-astra"}}}}"#
            ),
            tc(&t, 1000, 900, 800),
            tc(&t2, 1000, 900, 800),
            format!(
                r#"{{"timestamp":"{t2}","type":"event_msg","payload":{{"type":"token_count","info":null}}}}"#
            ),
            tc(&t2, 1500, 450, 400),
        ],
    );

    let out = usage(&home);

    // replies, input, output, cache read, cache write
    assert_eq!(
        row(&out, "pi", "cliproxyapi:developer")[2..],
        ["claude", "2", "300", "20", "2.0k", "0"],
        "{out}"
    );
    assert_eq!(
        row(&out, "pi", "cliproxyapi")[2..],
        ["gpt", "1", "7", "10", "1.0k", "0"],
        "{out}"
    );
    assert_eq!(
        row(&out, "claude", "~/.claude")[2..],
        ["claude", "1", "3", "50", "400", "60"],
        "{out}"
    );
    assert_eq!(
        row(&out, "codex", "openai")[2..],
        ["gpt", "2", "150", "10", "1.2k", "0"],
        "{out}"
    );
    assert!(
        out.contains("~/.claude is signed in to dev@example.org"),
        "{out}"
    );
    assert!(
        out.contains("ravnori made no model calls of its own"),
        "{out}"
    );
}

#[test]
fn an_empty_home_says_so() {
    let base = ravnori::TempDir::new("usage-empty");
    let home = base.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let out = usage(&home);
    assert!(out.contains("No replies with token counts"), "{out}");
}
