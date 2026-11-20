//! Replaced memories: a newer memory that makes an older one outdated on the same point.
//!
//! Distillation adds memories and never changes them, so a corrected finding, a reversed
//! decision or a fixed bug leaves the older memory beside the newer one. On 30 days of
//! real use (20 Nov 2026), 33% of the memories ravnori offered to agents (95% interval
//! 25-42%, 115 sampled) had already been made outdated by a newer memory, as two models
//! from different families judged it; for prompt recall, half.
//!
//! Each new observation is compared with the older observations of its project that are
//! closest in meaning (at most three, written in the 30 days before it), and the
//! distillation model decides, one call per new memory, which of them it replaces. Being
//! similar is not enough: among close pairs only 15-28% were replacements at any
//! similarity, most add detail to a memory that stays true.
//!
//! A newer memory that says the same as an older one is linked too: a status repeated
//! across sessions ("still blocked", then "still blocked") then leads to whatever replaced
//! the last of them ("fix landed"), which the first one was never compared with.
//!
//! Nothing is changed or deleted. Prompt recall shows the newest memory in place of the
//! ones it replaced or repeated, the session start leaves those out, and search, `rvn
//! ask` and get_observations show both, linked. `recall.hide_replaced: false` shows both
//! everywhere, as before.

use crate::config::CONFIG;
use crate::db;
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

/// What the model is asked. On 224 pairs judged one new memory per call, this wording
/// marked 53 replacements, 43 of them confirmed by at least one of two models judging
/// each pair alone; a stricter wording ("name the statement NEW makes false") lowered
/// both precision and recall.
pub const SYSTEM: &str = r#"Memories from a developer's coding-agent sessions. Each case is one NEW memory and OLD memories from the same project written before it.
For each OLD memory, decide what its NEW memory does to it, for a future agent that might be shown OLD alone:
- "supersedes": NEW makes OLD wrong or outdated on the same point: a corrected finding, a reversed or changed decision, a bug or blocker OLD says is open that NEW says is resolved, a status that moved on, a value or behaviour that changed. Showing OLD without NEW would mislead.
- "extends": same topic; NEW adds detail or a later step, and what OLD says is still true.
- "duplicate": NEW says the same as OLD.
- "unrelated": different topics, or only loosely related.
When unsure between "supersedes" and "extends", answer "extends".
Return JSON only: {"verdicts": [{"old": 123, "new": 456, "verdict": "...", "reason": "one short sentence"}]} with one entry per OLD memory of every case."#;

/// Older memories below this cosine to the new one are not compared: of the sampled
/// pairs under it, none was judged a replacement.
pub const MIN_COSINE: f32 = 0.55;
/// Compared with at most this many older memories, the closest first. One call either
/// way; with three, a fix was never compared with the sixth-closest memory that still
/// called it open (20 Nov 2026).
pub const CANDIDATES: usize = 5;
/// How far back an older memory may be.
pub const WINDOW_DAYS: i64 = 30;
/// Model calls the watcher spends on this in 24 hours unless `distill.supersede_calls`
/// says otherwise; counted apart from distillation's own limit, so memories are never
/// made later because of it.
pub const DAILY_CALLS: usize = 200;
/// The `distill_calls` source these calls are recorded under.
pub const SOURCE: &str = "supersede";

const DAY_MS: i64 = 86_400_000;

/// Whether recall acts on replacements (`recall.hide_replaced`, default true).
pub fn enabled() -> bool {
    CONFIG.recall.hide_replaced != Some(false)
}

pub fn daily_calls() -> usize {
    CONFIG.distill.supersede_calls.unwrap_or(DAILY_CALLS)
}

/// A replacement recall can show in the older memory's place: an observation from a
/// session that is not scripted, not a personal detail, written before `before`.
fn usable(alias: &str, before: i64) -> String {
    format!(
        "{alias}.kind = 'observation' AND coalesce({alias}.type, '') != 'sensitive'
         AND coalesce({alias}.created_at, 0) < {before}
         AND NOT EXISTS (SELECT 1 FROM scripted_sessions x WHERE x.session_id = {alias}.session_id)"
    )
}

/// For a query over `memories m`: m has not been replaced by a memory recall could show
/// instead, written before `before` (a replayed prompt sees only what existed then).
/// Always true when `recall.hide_replaced` is false.
pub fn not_replaced(before: Option<i64>) -> String {
    if !enabled() {
        return "1".into();
    }
    format!(
        "NOT EXISTS (SELECT 1 FROM supersedes r JOIN memories n ON n.id = r.new_id
                      WHERE r.old_id = m.id AND {})",
        usable("n", before.unwrap_or(i64::MAX))
    )
}

/// The newest memory that replaced `id`, following replacements of replacements, among
/// those recall could show, written before `before`. None when nothing replaced it (or
/// `recall.hide_replaced` is false).
pub fn latest(conn: &Connection, id: i64, before: Option<i64>) -> Result<Option<i64>> {
    if !enabled() {
        return Ok(None);
    }
    let mut st = conn.prepare_cached(&format!(
        "SELECT n.id FROM supersedes r JOIN memories n ON n.id = r.new_id
          WHERE r.old_id = ?1 AND {}
          ORDER BY n.created_at DESC, n.id DESC LIMIT 1",
        usable("n", before.unwrap_or(i64::MAX))
    ))?;
    let mut seen = vec![id];
    let mut cur = id;
    while let Some(next) = st.query_row([cur], |r| r.get::<_, i64>(0)).optional()? {
        if seen.contains(&next) || seen.len() > 20 {
            break;
        }
        seen.push(next);
        cur = next;
    }
    Ok((cur != id).then_some(cur))
}

/// One link as shown to people and agents.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    /// The other memory: the newer one for `replaced_by`, the older one for `replaces`.
    pub id: i64,
    pub at: i64,
    pub title: String,
    pub reason: String,
    /// "supersedes" or "duplicate".
    pub relation: String,
}

impl Link {
    /// How the newer memory stands to the older one, for "#old <verb> #new".
    pub fn verb(&self) -> &'static str {
        if self.relation == "duplicate" {
            "repeated in"
        } else {
            "replaced by"
        }
    }
}

fn links(conn: &Connection, sql: &str, id: i64) -> Result<Vec<Link>> {
    let mut st = conn.prepare_cached(sql)?;
    let rows = st
        .query_map([id], |r| {
            Ok(Link {
                id: r.get(0)?,
                at: r.get(1)?,
                title: r.get(2)?,
                reason: r.get(3)?,
                relation: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

/// The newer memories that replaced `id`, newest first.
pub fn replaced_by(conn: &Connection, id: i64) -> Result<Vec<Link>> {
    links(
        conn,
        "SELECT n.id, coalesce(n.created_at, 0), coalesce(n.title, ''), coalesce(r.reason, ''), r.relation
           FROM supersedes r JOIN memories n ON n.id = r.new_id
          WHERE r.old_id = ?1 ORDER BY n.created_at DESC, n.id DESC",
        id,
    )
}

/// The older memories `id` replaced, newest first.
pub fn replaces(conn: &Connection, id: i64) -> Result<Vec<Link>> {
    links(
        conn,
        "SELECT o.id, coalesce(o.created_at, 0), coalesce(o.title, ''), coalesce(r.reason, ''), r.relation
           FROM supersedes r JOIN memories o ON o.id = r.old_id
          WHERE r.new_id = ?1 ORDER BY o.created_at DESC, o.id DESC",
        id,
    )
}

// ---------------------------------------------------------------- the check

/// The vector model whose vectors are compared: the configured one, under the key most
/// of its vectors carry (name plus a fingerprint of its files).
fn vector_key(conn: &Connection) -> Result<Option<String>> {
    let name = crate::embed::model_name();
    Ok(conn
        .query_row(
            "SELECT model FROM memory_vectors WHERE model = ?1 OR model LIKE ?1 || '@%'
              GROUP BY model ORDER BY count(*) DESC LIMIT 1",
            [&name],
            |r| r.get(0),
        )
        .optional()?)
}

/// A stored vector, back to unit length.
fn unit(scale: f32, q: &[u8]) -> Vec<f32> {
    let v: Vec<f32> = q.iter().map(|b| f32::from(*b as i8) * scale).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.into_iter().map(|x| x / n).collect()
    } else {
        v
    }
}

/// The older observations of `id`'s project closest in meaning to it, best first: at
/// most CANDIDATES, at least MIN_COSINE, written in the WINDOW_DAYS before it.
pub fn candidates(conn: &Connection, key: &str, id: i64) -> Result<Vec<(i64, f32)>> {
    let me: Option<(String, i64, f32, Vec<u8>)> = conn
        .query_row(
            "SELECT m.project, m.created_at, v.scale, v.vec FROM memories m
               JOIN memory_vectors v ON v.memory_id = m.id AND v.model = ?2
              WHERE m.id = ?1 AND m.project IS NOT NULL AND m.created_at IS NOT NULL",
            params![id, key],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let Some((project, at, scale, vec)) = me else {
        return Ok(Vec::new());
    };
    let q = unit(scale, &vec);
    let mut st = conn.prepare_cached(&format!(
        "SELECT m.id, v.scale, v.vec FROM memories m
           JOIN memory_vectors v ON v.memory_id = m.id AND v.model = ?2
          WHERE m.project = ?1 AND m.kind = 'observation' AND coalesce(m.type, '') != 'sensitive'
            AND (m.created_at < ?3 OR (m.created_at = ?3 AND m.id < ?4)) AND m.created_at >= ?5
            AND {}",
        crate::scripted::MEMORY_NOT_SCRIPTED
    ))?;
    let mut scored: Vec<(i64, f32)> = st
        .query_map(
            params![project, key, at, id, at - WINDOW_DAYS * DAY_MS],
            |r| {
                let (oid, s, v): (i64, f32, Vec<u8>) = (r.get(0)?, r.get(1)?, r.get(2)?);
                Ok((oid, s, v))
            },
        )?
        .filter_map(Result::ok)
        .filter(|(_, _, v)| v.len() == q.len())
        .map(|(oid, s, v)| {
            let u = unit(s, &v);
            (oid, q.iter().zip(&u).map(|(a, b)| a * b).sum::<f32>())
        })
        .filter(|(_, c)| *c >= MIN_COSINE)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(CANDIDATES);
    Ok(scored)
}

/// The text of a memory as the model is shown it.
fn memory_text(conn: &Connection, id: i64) -> Result<String> {
    Ok(conn.query_row(
        "SELECT coalesce(title, ''), coalesce(subtitle, ''), coalesce(narrative, ''),
                coalesce(facts, ''), date(coalesce(created_at, 0) / 1000, 'unixepoch')
           FROM memories WHERE id = ?1",
        [id],
        |r| {
            Ok(format!(
                "date: {}\ntitle: {}\nsubtitle: {}\nnarrative: {}\nfacts: {}",
                r.get::<_, String>(4)?,
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                crate::text::head(&r.get::<_, String>(2)?, 1500),
                crate::text::head(&r.get::<_, String>(3)?, 1500),
            ))
        },
    )?)
}

/// The case sent to the model for new memory `id` and its older candidates.
pub fn case(conn: &Connection, id: i64, olds: &[i64]) -> Result<String> {
    let mut w = format!("=== CASE: NEW #{id}\n{}\n", memory_text(conn, id)?);
    for o in olds {
        w.push_str(&format!("--- OLD #{o}\n{}\n", memory_text(conn, *o)?));
    }
    Ok(w)
}

fn as_id(v: &Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_str()?.trim().trim_start_matches('#').parse().ok())
}

/// One older memory a new one replaced or repeated.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub old: i64,
    /// "supersedes" or "duplicate".
    pub relation: &'static str,
    pub reason: String,
}

/// The older memories the model says `new` replaces or repeats, with its reason: only ids
/// it was shown as candidates, and only verdicts about `new` (an answer about another
/// memory or an id it was not shown is ignored).
pub fn read_verdicts(v: &Value, new: i64, olds: &[i64]) -> Vec<Verdict> {
    let mut out: Vec<Verdict> = Vec::new();
    for x in v["verdicts"].as_array().into_iter().flatten() {
        let Some(old) = x.get("old").and_then(as_id) else {
            continue;
        };
        let about_new = x.get("new").and_then(as_id).is_none_or(|n| n == new);
        let relation = match x["verdict"].as_str().map(str::trim) {
            Some("supersedes") => "supersedes",
            Some("duplicate") => "duplicate",
            _ => continue,
        };
        if about_new && olds.contains(&old) && !out.iter().any(|f| f.old == old) {
            let reason = crate::text::redact(x["reason"].as_str().unwrap_or("").trim());
            out.push(Verdict {
                old,
                relation,
                reason: crate::text::head(&reason, 300).to_string(),
            });
        }
    }
    out
}

/// Record that `new` was compared, and the replacements found, in one write. A memory
/// forgotten while the model answered is left alone.
pub fn save(
    conn: &mut Connection,
    new: i64,
    candidates: usize,
    found: &[Verdict],
    model: Option<&str>,
) -> Result<usize> {
    let tx = conn.transaction()?;
    let alive = |id: i64| -> rusqlite::Result<bool> {
        tx.query_row("SELECT count(*) FROM memories WHERE id = ?1", [id], |r| {
            r.get::<_, i64>(0)
        })
        .map(|n| n > 0)
    };
    if !alive(new)? {
        return Ok(0);
    }
    let mut n = 0;
    for f in found {
        if alive(f.old)? {
            n += tx.execute(
                "INSERT OR IGNORE INTO supersedes(old_id, new_id, relation, reason, model, at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![f.old, new, f.relation, f.reason, model, db::now_ms()],
            )?;
        }
    }
    tx.execute(
        "INSERT INTO supersede_checked(memory_id, candidates, at) VALUES (?1, ?2, ?3)
         ON CONFLICT(memory_id) DO UPDATE SET candidates = excluded.candidates, at = excluded.at",
        params![new, candidates as i64, db::now_ms()],
    )?;
    tx.commit()?;
    Ok(n)
}

/// Observations of the last `since_days` not compared yet that have a vector, newest
/// first (what recall meets soonest).
fn pending(conn: &Connection, key: &str, since_days: i64, limit: usize) -> Result<Vec<i64>> {
    let mut st = conn.prepare(&format!(
        "SELECT m.id FROM memories m
           JOIN memory_vectors v ON v.memory_id = m.id AND v.model = ?1
          WHERE m.kind = 'observation' AND coalesce(m.type, '') != 'sensitive'
            AND m.created_at >= ?2 AND m.project IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM supersede_checked c WHERE c.memory_id = m.id)
            AND {}
          ORDER BY m.created_at DESC, m.id DESC LIMIT ?3",
        crate::scripted::MEMORY_NOT_SCRIPTED
    ))?;
    let ids = st
        .query_map(
            params![key, db::now_ms() - since_days * DAY_MS, limit as i64],
            |r| r.get(0),
        )?
        .collect::<rusqlite::Result<_>>()?;
    Ok(ids)
}

/// Requests these checks sent in the last 24 hours.
pub fn calls_last_day(conn: &Connection) -> Result<usize> {
    Ok(conn.query_row(
        "SELECT coalesce(sum(requests), 0) FROM distill_calls WHERE at > ?1 AND source = ?2",
        params![db::now_ms() - DAY_MS, SOURCE],
        |r| r.get::<_, i64>(0),
    )? as usize)
}

pub struct Options {
    pub since_days: i64,
    /// Stop after this many model calls.
    pub max_calls: Option<usize>,
    /// Stop once these checks sent this many requests in the last 24 hours.
    pub budget: Option<usize>,
    /// Count what would be sent, without calling the model.
    pub dry_run: bool,
}

#[derive(Debug, Default)]
pub struct Stats {
    /// Memories compared (with or without a model call).
    pub checked: usize,
    /// Of those, with no older memory close enough: no call needed.
    pub nothing_close: usize,
    pub calls: usize,
    /// Replacements recorded.
    pub links: usize,
    /// Dry run: memories that would need a call.
    pub would_call: usize,
    pub errors: Vec<String>,
}

/// Compare new memories with the older ones closest to them, newest first.
pub fn run(conn: &mut Connection, o: &Options) -> Result<Stats> {
    let mut st = Stats::default();
    let Some(_lock) = crate::distill::Lock::acquire_named(conn, "supersede.lock")? else {
        st.errors.push("another check is running".into());
        return Ok(st);
    };
    let Some(key) = vector_key(conn)? else {
        // No vectors yet (semantic recall off, or the model not loaded): nothing to compare.
        return Ok(st);
    };
    let mut llm: Option<crate::distill::Llm> = None;
    loop {
        // A dry run writes nothing, so it reads every pending memory at once.
        let batch = pending(
            conn,
            &key,
            o.since_days,
            if o.dry_run { usize::MAX >> 1 } else { 50 },
        )?;
        if batch.is_empty() {
            break;
        }
        let mut progressed = false;
        for id in batch {
            if o.max_calls
                .is_some_and(|m| st.calls.max(st.would_call) >= m)
            {
                return finish(conn, llm, st);
            }
            let close = candidates(conn, &key, id)?;
            if close.is_empty() {
                if !o.dry_run {
                    save(conn, id, 0, &[], None)?;
                }
                st.checked += 1;
                st.nothing_close += 1;
                progressed = true;
                continue;
            }
            if o.dry_run {
                st.would_call += 1;
                continue;
            }
            let left = match o.budget {
                Some(b) => match b.saturating_sub(calls_last_day(conn)?) {
                    0 => return finish(conn, llm, st),
                    n => Some(n),
                },
                None => None,
            };
            if llm.is_none() {
                let l = crate::distill::Llm::from_config()?;
                l.load_cooldowns(conn);
                llm = Some(l);
            }
            let l = llm.as_ref().expect("loaded above");
            let olds: Vec<i64> = close.iter().map(|c| c.0).collect();
            let sent = l.sent();
            l.limit_to(left.map(|n| sent + n));
            st.calls += 1;
            let result = l.ask(SYSTEM, &case(conn, id, &olds)?);
            crate::distill::record_call(
                conn,
                SOURCE,
                result.as_ref().ok().map(|(_, m)| m.as_str()),
                result.is_ok(),
                l.sent() - sent,
            )?;
            match result {
                Ok((v, model)) => {
                    let found = read_verdicts(&v, id, &olds);
                    st.links += save(conn, id, olds.len(), &found, Some(&model))?;
                    st.checked += 1;
                    progressed = true;
                }
                Err(e) => {
                    // Left unchecked: the next pass tries it again.
                    st.errors.push(format!("#{id}: {e:#}"));
                    return finish(conn, llm, st);
                }
            }
        }
        if o.dry_run || !progressed {
            break;
        }
    }
    finish(conn, llm, st)
}

fn finish(conn: &Connection, llm: Option<crate::distill::Llm>, st: Stats) -> Result<Stats> {
    if let Some(l) = &llm {
        l.save_cooldowns(conn)?;
    }
    Ok(st)
}

/// The watcher's share: a few calls per pass, within `distill.supersede_calls` a day
/// (0 turns the checks off; recall still uses the links already made).
pub fn watch_pass(conn: &mut Connection) {
    let budget = daily_calls();
    if budget == 0 || !enabled() || !crate::recall::semantic_enabled() {
        return;
    }
    let o = Options {
        since_days: WINDOW_DAYS,
        max_calls: Some(10),
        budget: Some(budget),
        dry_run: false,
    };
    match run(conn, &o) {
        Ok(s) if s.calls > 0 || !s.errors.is_empty() => crate::hook::log(&format!(
            "watch supersede: {} checked, {} calls, {} replaced, {} errors",
            s.checked,
            s.calls,
            s.links,
            s.errors.len()
        )),
        Ok(_) => {}
        Err(e) => crate::hook::log(&format!("watch supersede: {e:#}")),
    }
}

/// Where the checks stand, for doctor.
pub struct Status {
    pub links: usize,
    pub replaced: usize,
    pub calls_day: usize,
    pub budget: usize,
    /// Observations of the last WINDOW_DAYS not compared yet.
    pub unchecked: usize,
}

pub fn status(conn: &Connection) -> Result<Status> {
    let (links, replaced): (i64, i64) = conn.query_row(
        "SELECT count(*), count(DISTINCT old_id) FROM supersedes",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let unchecked: i64 = conn.query_row(
        &format!(
            "SELECT count(*) FROM memories m
              WHERE m.kind = 'observation' AND coalesce(m.type, '') != 'sensitive'
                AND m.created_at >= ?1 AND m.project IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM supersede_checked c WHERE c.memory_id = m.id)
                AND {}",
            crate::scripted::MEMORY_NOT_SCRIPTED
        ),
        [db::now_ms() - WINDOW_DAYS * DAY_MS],
        |r| r.get(0),
    )?;
    Ok(Status {
        links: links as usize,
        replaced: replaced as usize,
        calls_day: calls_last_day(conn)?,
        budget: daily_calls(),
        unchecked: unchecked as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn db() -> Connection {
        db::open_with(
            std::path::Path::new(":memory:"),
            std::time::Duration::from_secs(1),
        )
        .unwrap()
    }

    fn obs(c: &Connection, id: i64, at: i64, title: &str) {
        c.execute(
            "INSERT INTO memories(id, session_id, project, kind, type, title, origin, origin_id, created_at)
             VALUES (?1, 'pi:s', 'p', 'observation', 'discovery', ?2, 'ravnori', ?1, ?3)",
            params![id, title, at],
        )
        .unwrap();
    }

    fn link(c: &Connection, old: i64, new: i64) {
        c.execute(
            "INSERT INTO supersedes(old_id, new_id, reason, at) VALUES (?1, ?2, 'r', 0)",
            params![old, new],
        )
        .unwrap();
    }

    #[test]
    fn verdicts_keep_only_replacements_of_shown_candidates_about_this_memory() {
        let v = json!({"verdicts": [
            {"old": 1, "new": 9, "verdict": "supersedes", "reason": "fixed since"},
            {"old": "#2", "new": "#9", "verdict": " supersedes ", "reason": "moved on"},
            {"old": 3, "new": 9, "verdict": "extends"},
            {"old": 5, "new": 9, "verdict": "duplicate", "reason": "same status"},
            {"old": 6, "new": 9, "verdict": "unrelated"},
            {"old": 4, "new": 9, "verdict": "supersedes"},
            {"old": 1, "new": 8, "verdict": "supersedes"},
            {"old": 1, "new": 9, "verdict": "supersedes", "reason": "again"},
            {"verdict": "supersedes"}
        ]});
        let got: Vec<(i64, &str, String)> = read_verdicts(&v, 9, &[1, 2, 3, 5, 6])
            .into_iter()
            .map(|f| (f.old, f.relation, f.reason))
            .collect();
        assert_eq!(
            got,
            vec![
                (1, "supersedes", "fixed since".to_string()),
                (2, "supersedes", "moved on".to_string()),
                (5, "duplicate", "same status".to_string()),
            ]
        );
        assert!(read_verdicts(&json!({"verdicts": "no"}), 9, &[1]).is_empty());
    }

    #[test]
    fn the_latest_replacement_is_followed_through_a_chain_and_stops_at_cycles() {
        let c = db();
        for (id, at) in [(1, 10), (2, 20), (3, 30), (4, 40)] {
            obs(&c, id, at, "t");
        }
        link(&c, 1, 2);
        link(&c, 2, 3);
        assert_eq!(latest(&c, 1, None).unwrap(), Some(3));
        assert_eq!(latest(&c, 2, None).unwrap(), Some(3));
        assert_eq!(latest(&c, 3, None).unwrap(), None);
        // A replayed prompt only sees replacements written before it.
        assert_eq!(latest(&c, 1, Some(25)).unwrap(), Some(2));
        assert_eq!(latest(&c, 1, Some(15)).unwrap(), None);
        // A bad cycle (never written by the check, which only links newer to older)
        // does not loop.
        link(&c, 3, 1);
        assert!(latest(&c, 1, None).unwrap().is_some());
    }

    #[test]
    fn a_repeated_status_leads_to_what_replaced_its_last_repeat() {
        let c = db();
        for (id, at) in [(1, 10), (2, 20), (3, 30)] {
            obs(&c, id, at, "t");
        }
        // "still blocked" (1), said again (2), then "fix landed" (3), compared with 2 only.
        c.execute(
            "INSERT INTO supersedes(old_id, new_id, relation, at) VALUES (1, 2, 'duplicate', 0), (2, 3, 'supersedes', 0)",
            [],
        )
        .unwrap();
        assert_eq!(latest(&c, 1, None).unwrap(), Some(3));
        assert_eq!(replaced_by(&c, 1).unwrap()[0].verb(), "repeated in");
        assert_eq!(replaced_by(&c, 2).unwrap()[0].verb(), "replaced by");
    }

    #[test]
    fn a_replacement_recall_could_not_show_does_not_hide_the_older_memory() {
        let c = db();
        obs(&c, 1, 10, "old");
        obs(&c, 2, 20, "new but personal");
        c.execute("UPDATE memories SET type = 'sensitive' WHERE id = 2", [])
            .unwrap();
        link(&c, 1, 2);
        assert_eq!(latest(&c, 1, None).unwrap(), None);
        let hidden: i64 = c
            .query_row(
                &format!(
                    "SELECT count(*) FROM memories m WHERE m.id = 1 AND {}",
                    not_replaced(None)
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hidden, 1, "still shown");
        c.execute("UPDATE memories SET type = 'decision' WHERE id = 2", [])
            .unwrap();
        let shown: i64 = c
            .query_row(
                &format!(
                    "SELECT count(*) FROM memories m WHERE m.id = 1 AND {}",
                    not_replaced(None)
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(shown, 0, "replaced");
    }

    #[test]
    fn links_go_with_a_forgotten_memory() {
        let mut c = db();
        obs(&c, 1, 10, "old");
        obs(&c, 2, 20, "new");
        let v = |old: i64, reason: &str| Verdict {
            old,
            relation: "supersedes",
            reason: reason.into(),
        };
        save(&mut c, 2, 1, &[v(1, "moved on")], Some("m")).unwrap();
        assert_eq!(replaced_by(&c, 1).unwrap()[0].id, 2);
        assert_eq!(replaces(&c, 2).unwrap()[0].reason, "moved on");
        crate::forget::forget(&mut c, &["2".to_string()], None, None).unwrap();
        let left: (i64, i64) = c
            .query_row(
                "SELECT (SELECT count(*) FROM supersedes), (SELECT count(*) FROM supersede_checked)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(left, (0, 0));
        // A memory gone while the model answered gets no link.
        assert_eq!(save(&mut c, 1, 1, &[v(99, "x")], Some("m")).unwrap(), 0);
    }

    #[test]
    fn candidates_are_the_closest_older_memories_of_the_project_in_the_window() {
        let c = db();
        let key = "test@1";
        let put = |id: i64, at: i64, project: &str, v: [f32; 2]| {
            c.execute(
                "INSERT INTO memories(id, session_id, project, kind, type, title, origin, origin_id, created_at)
                 VALUES (?1, 'pi:s', ?2, 'observation', 'discovery', 't', 'ravnori', ?1, ?3)",
                params![id, project, at],
            )
            .unwrap();
            let (scale, q) = crate::embed::quantize(&v);
            c.execute(
                "INSERT INTO memory_vectors(memory_id, model, dim, scale, vec, text_hash) VALUES (?1, ?2, 2, ?3, ?4, 'h')",
                params![id, key, scale, q],
            )
            .unwrap();
        };
        let day = DAY_MS;
        put(10, 40 * day, "p", [1.0, 0.0]); // the new memory
        put(1, 39 * day, "p", [0.99, 0.14]); // close, older
        put(2, 38 * day, "p", [0.8, 0.6]); // close enough (0.8)
        put(3, 37 * day, "p", [0.0, 1.0]); // far
        put(4, 5 * day, "p", [1.0, 0.0]); // identical but outside 30 days
        put(5, 39 * day, "q", [1.0, 0.0]); // another project
        put(6, 41 * day, "p", [1.0, 0.0]); // newer
        let got: Vec<i64> = candidates(&c, key, 10)
            .unwrap()
            .into_iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(got, vec![1, 2]);
    }
}
