//! Token usage by week, read from the agents' own transcripts.
//!
//! Every assistant reply in a Claude Code, Codex or pi transcript carries the tokens the
//! provider reported for it. `rvn usage` adds them up per week, agent, route (the Claude
//! Code profile, Codex's provider, pi's provider and model prefix such as a CLIProxyAPI
//! account) and model family, so a change in plan usage can be traced to where the tokens
//! went instead of guessed. Nothing is stored: transcripts are read where they are.
//!
//! A reply can appear more than once: Claude Code writes one line per streamed content
//! block, Codex repeats a token count when nothing new was used, and a forked or resumed
//! session copies its history into a new file. Each reply is counted once, by the id its
//! provider gave it.

use crate::adapters::str_of;
use crate::model::Agent;
use anyhow::Result;
use rayon::prelude::*;
use rusqlite::Connection;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const DAY: i64 = 86_400_000;

/// Tokens as providers report them. `input` is input not read from a cache.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokens {
    pub replies: u64,
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

impl Tokens {
    fn add(&mut self, o: &Tokens) {
        self.replies += o.replies;
        self.input += o.input;
        self.output += o.output;
        self.cache_read += o.cache_read;
        self.cache_write += o.cache_write;
    }

    fn is_empty(&self) -> bool {
        self.input + self.output + self.cache_read + self.cache_write == 0
    }
}

/// Where a week's tokens went.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bucket {
    /// Local day number (days since 1970-01-01) of the week's Monday.
    pub week: i64,
    pub agent: &'static str,
    pub route: String,
    pub family: &'static str,
}

/// One reply's usage, as its transcript recorded it.
#[derive(Debug)]
struct Reply {
    at: i64,
    /// The provider's id for the reply, to count copies once.
    key: Option<String>,
    route: String,
    model: String,
    tokens: Tokens,
}

/// The local day number of the Monday starting the week that holds `at`.
pub fn monday(at: i64, offset_min: i64) -> i64 {
    let day = (at + offset_min * 60_000).div_euclid(DAY);
    // 1970-01-01 was a Thursday: three days after a Monday.
    day - (day + 3).rem_euclid(7)
}

/// Epoch ms when the local day `day` starts.
fn day_start(day: i64, offset_min: i64) -> i64 {
    day * DAY - offset_min * 60_000
}

/// The start of the window holding this week and the `weeks - 1` before it.
pub fn window_start(now: i64, weeks: i64, offset_min: i64) -> i64 {
    day_start(monday(now, offset_min) - 7 * (weeks.max(1) - 1), offset_min)
}

fn family(model: &str) -> &'static str {
    let m = model.to_ascii_lowercase();
    if m.contains("claude") {
        "claude"
    } else if m.contains("gpt") || m.contains("codex") {
        "gpt"
    } else if m.contains("gemini") {
        "gemini"
    } else {
        "other"
    }
}

fn n(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::as_u64).unwrap_or(0)
}

fn ts(v: &Value) -> Option<i64> {
    crate::text::parse_ts(str_of(v, "timestamp")?)
}

/// pi: `{"type":"message","message":{"role":"assistant","provider","model","usage",...}}`.
fn pi_reply(v: &Value) -> Option<Reply> {
    if str_of(v, "type") != Some("message") {
        return None;
    }
    let m = v.get("message")?;
    if str_of(m, "role") != Some("assistant") {
        return None;
    }
    let u = m.get("usage")?;
    let at = m
        .get("timestamp")
        .and_then(Value::as_i64)
        .or_else(|| ts(v))?;
    let provider = str_of(m, "provider").unwrap_or("-");
    let full = str_of(m, "model").unwrap_or("?");
    // A proxy names the account or upstream before a slash: "developer/claude-opus-5-5".
    let (route, model) = match full.split_once('/') {
        Some((prefix, model)) => (format!("{provider}:{prefix}"), model),
        None => (provider.to_string(), full),
    };
    Some(Reply {
        at,
        key: str_of(m, "responseId").map(String::from),
        route,
        model: model.to_string(),
        tokens: Tokens {
            replies: 1,
            input: n(u, "input"),
            output: n(u, "output"),
            cache_read: n(u, "cacheRead"),
            cache_write: n(u, "cacheWrite"),
        },
    })
}

/// Claude Code: `{"type":"assistant","timestamp","message":{"id","model","usage"}}`.
fn claude_reply(v: &Value, route: &str) -> Option<Reply> {
    if str_of(v, "type") != Some("assistant") {
        return None;
    }
    let m = v.get("message")?;
    let u = m.get("usage")?;
    let model = str_of(m, "model").unwrap_or("?");
    if model == "<synthetic>" {
        return None;
    }
    Some(Reply {
        at: ts(v)?,
        key: str_of(m, "id")
            .or_else(|| str_of(v, "requestId"))
            .map(String::from),
        route: route.to_string(),
        model: model.to_string(),
        tokens: Tokens {
            replies: 1,
            input: n(u, "input_tokens"),
            output: n(u, "output_tokens"),
            cache_read: n(u, "cache_read_input_tokens"),
            cache_write: n(u, "cache_creation_input_tokens"),
        },
    })
}

/// Codex keeps provider and model in earlier records than the token counts.
#[derive(Default)]
struct CodexState {
    provider: Option<String>,
    model: Option<String>,
    last_total: Option<u64>,
}

/// Codex: `session_meta` (provider), `turn_context` (model), then `event_msg` token counts
/// with the latest call's usage. OpenAI's input count includes cached input.
fn codex_reply(v: &Value, st: &mut CodexState) -> Option<Reply> {
    let p = v.get("payload")?;
    match str_of(v, "type")? {
        "session_meta" => {
            st.provider = str_of(p, "model_provider").map(String::from);
            None
        }
        "turn_context" => {
            st.model = str_of(p, "model").map(String::from);
            None
        }
        "event_msg" if str_of(p, "type") == Some("token_count") => {
            let info = p.get("info").filter(|i| !i.is_null())?;
            let total = info.get("total_token_usage").map(|t| n(t, "total_tokens"));
            // The same total again: nothing was used since the last count.
            if total.is_some() && total == st.last_total {
                return None;
            }
            st.last_total = total;
            let u = info.get("last_token_usage")?;
            let stamp = str_of(v, "timestamp").unwrap_or_default();
            let cached = n(u, "cached_input_tokens");
            Some(Reply {
                at: ts(v)?,
                key: Some(format!("{stamp}:{}", total.unwrap_or(0))),
                route: st.provider.clone().unwrap_or_else(|| "-".into()),
                model: st.model.clone().unwrap_or_else(|| "?".into()),
                tokens: Tokens {
                    replies: 1,
                    input: n(u, "input_tokens").saturating_sub(cached),
                    output: n(u, "output_tokens"),
                    cache_read: cached,
                    cache_write: n(u, "cache_write_input_tokens"),
                },
            })
        }
        _ => None,
    }
}

/// `~/.claude` for a Claude Code projects root, or the claude-mem observer for its runs.
fn claude_route(root: &Path, file: &Path) -> String {
    if file
        .to_string_lossy()
        .contains("claude-mem-observer-sessions")
    {
        return "claude-mem observer".into();
    }
    tilde(root.parent().unwrap_or(root))
}

fn tilde(p: &Path) -> String {
    match p.strip_prefix(crate::db::home()) {
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => p.display().to_string(),
    }
}

/// Replies in one transcript at or after `from`. Unreadable lines are skipped: a
/// transcript being written ends in a partial line.
fn read_file(path: &Path, root: &Path, agent: Agent, from: i64) -> Vec<Reply> {
    let Ok(f) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let route = claude_route(root, path);
    let mut codex = CodexState::default();
    let mut out = Vec::new();
    let mut buf = Vec::new();
    let mut r = BufReader::new(f);
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf);
        let wanted = match agent {
            Agent::Pi | Agent::Claude => line.contains("\"usage\""),
            Agent::Codex => {
                line.contains("\"token_count\"")
                    || line.contains("\"session_meta\"")
                    || line.contains("\"turn_context\"")
            }
        };
        if !wanted {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let reply = match agent {
            Agent::Pi => pi_reply(&v),
            Agent::Claude => claude_reply(&v, &route),
            Agent::Codex => codex_reply(&v, &mut codex),
        };
        if let Some(rp) = reply.filter(|rp| rp.at >= from && !rp.tokens.is_empty()) {
            out.push(rp);
        }
    }
    out
}

/// Tokens per week, agent, route and model family since `from`, from every transcript
/// under `roots` written to since then.
pub fn collect(roots: &[(PathBuf, Agent)], from: i64, offset_min: i64) -> BTreeMap<Bucket, Tokens> {
    let from_time = std::time::UNIX_EPOCH + std::time::Duration::from_millis(from.max(0) as u64);
    let mut files = Vec::new();
    for (root, agent) in roots {
        for e in walkdir::WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let jsonl =
                e.file_type().is_file() && e.path().extension().is_some_and(|x| x == "jsonl");
            // A file last written before the window holds nothing in it.
            let recent = e
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .is_some_and(|t| t >= from_time);
            if jsonl && recent {
                files.push((e.into_path(), root.as_path(), *agent));
            }
        }
    }
    let per_file: Vec<(Agent, Vec<Reply>)> = files
        .par_iter()
        .map(|(path, root, agent)| (*agent, read_file(path, root, *agent, from)))
        .collect();
    let mut seen: HashSet<(Agent, String)> = HashSet::new();
    let mut out: BTreeMap<Bucket, Tokens> = BTreeMap::new();
    for (agent, replies) in per_file {
        for rp in replies {
            if let Some(k) = &rp.key
                && !seen.insert((agent, k.clone()))
            {
                continue;
            }
            let b = Bucket {
                week: monday(rp.at, offset_min),
                agent: agent.as_str(),
                route: rp.route,
                family: family(&rp.model),
            };
            out.entry(b).or_default().add(&rp.tokens);
        }
    }
    out
}

/// The account a Claude Code profile is signed in to now, from its `.claude.json`.
fn claude_account(profile: &Path) -> Option<String> {
    let mut candidates = vec![profile.join(".claude.json")];
    if profile == crate::db::home().join(".claude") {
        candidates.insert(0, crate::db::home().join(".claude.json"));
    }
    candidates.iter().find_map(|p| {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
        str_of(v.get("oauthAccount")?, "emailAddress").map(String::from)
    })
}

/// ravnori's own model calls since `from`, per model, busiest first.
fn own_calls(conn: &Connection, from: i64) -> Result<Vec<(String, i64)>> {
    let mut st = conn.prepare(
        "SELECT coalesce(model, '?'), sum(requests) FROM distill_calls WHERE at >= ?1
          GROUP BY 1 ORDER BY 2 DESC",
    )?;
    let rows = st
        .query_map([from], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// "12.3M"-style counts: k, M and B with one decimal.
pub fn short(n: u64) -> String {
    let f = n as f64;
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", f / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1}M", f / 1e6),
        _ => format!("{:.2}B", f / 1e9),
    }
}

fn week_label(day: i64) -> String {
    const MON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let (y, m, d) = crate::text::civil_from_days(day);
    format!("{d} {} {y}", MON[(m - 1) as usize])
}

pub fn render(rows: &BTreeMap<Bucket, Tokens>, weeks: i64) -> String {
    let mut w = format!(
        "Tokens per week (Monday to Sunday, local time) over the last {weeks} weeks, from the agents' own transcripts.\n"
    );
    if rows.is_empty() {
        w.push_str("No replies with token counts in these weeks.\n");
        return w;
    }
    let rw = rows.keys().map(|b| b.route.len()).max().unwrap_or(5).max(5);
    w.push_str(&format!(
        "{:<11}  {:<6} {:<rw$}  {:<6} {:>7} {:>8} {:>8} {:>10} {:>11}\n",
        "week of",
        "agent",
        "route",
        "models",
        "replies",
        "input",
        "output",
        "cache read",
        "cache write"
    ));
    let mut last_week = None;
    for (b, t) in rows {
        let week = if last_week == Some(b.week) {
            String::new()
        } else {
            week_label(b.week)
        };
        last_week = Some(b.week);
        w.push_str(&format!(
            "{:<11}  {:<6} {:<rw$}  {:<6} {:>7} {:>8} {:>8} {:>10} {:>11}\n",
            week,
            b.agent,
            b.route,
            b.family,
            t.replies,
            short(t.input),
            short(t.output),
            short(t.cache_read),
            short(t.cache_write)
        ));
    }
    w
}

/// Notes that keep the table from being over-read: which account each Claude Code
/// profile uses now, and ravnori's own model calls, which no transcript holds.
pub fn footer(conn: &Connection, from: i64) -> Result<String> {
    let mut w = String::from(
        "\ninput = input not read from a cache. How a subscription counts cached reads against its\nlimit is not in these files.\n",
    );
    for profile in crate::ingest::claude_config_dirs() {
        if !profile.join("projects").is_dir() {
            continue;
        }
        let who = claude_account(&profile).unwrap_or_else(|| "no account found".into());
        w.push_str(&format!(
            "Claude Code {} is signed in to {who} now; a transcript does not say which account served it.\n",
            tilde(&profile)
        ));
    }
    let calls = own_calls(conn, from)?;
    if calls.is_empty() {
        w.push_str("ravnori made no model calls of its own in these weeks.\n");
    } else {
        let list: Vec<String> = calls.iter().map(|(m, n)| format!("{m} {n}")).collect();
        w.push_str(&format!(
            "ravnori's own model calls (distillation, ask) in these weeks, not in the table: {}.\n",
            list.join(", ")
        ));
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIB: i64 = 7 * 60;

    #[test]
    fn weeks_start_on_local_monday() {
        // Monday 2026-11-02 00:30 WIB is Sunday 1 Nov 17:30 UTC.
        let mon = crate::text::parse_ts("2026-11-01T17:30:00Z").unwrap();
        assert_eq!(week_label(monday(mon, WIB)), "2 Nov 2026");
        assert_eq!(week_label(monday(mon, 0)), "26 Oct 2026");
        let sun = crate::text::parse_ts("2026-11-08T16:59:59Z").unwrap();
        assert_eq!(monday(sun, WIB), monday(mon, WIB));
        assert_eq!(window_start(sun, 1, WIB), mon - 30 * 60_000);
    }

    #[test]
    fn short_counts() {
        assert_eq!(short(999), "999");
        assert_eq!(short(18_300_000), "18.3M");
        assert_eq!(short(2_879_900_000), "2.88B");
    }

    #[test]
    fn families() {
        assert_eq!(family("claude-opus-5-5"), "claude");
        assert_eq!(family("gpt-5.6-luna"), "gpt");
        assert_eq!(family("codex-auto-review"), "gpt");
        assert_eq!(family("gemini-3-pro"), "gemini");
        assert_eq!(family("glm-5"), "other");
    }
}
