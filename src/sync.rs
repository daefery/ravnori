//! Team memory (test phase): each teammate's distilled memories for a project travel as
//! compressed batches through a shared transport, and every machine adds the others' to
//! its own local memory, where search and recall work as for its own.
//!
//! - Only distilled memories go: observations and summaries made or imported here. Never
//!   transcripts, vectors, pins, or anything marked sensitive; text is redacted again
//!   before it leaves.
//! - Each author appends only to their own stream, so writers never conflict. An append
//!   is durable before anything is published, so a sync that cannot reach the remote
//!   sends the same batches next time.
//! - Incoming memories are stored with origin `team:<author>`: never sent on again, shown
//!   with their author, forgettable like any other (a forgotten one stays out). Only
//!   observations and summaries are accepted, so a teammate cannot add pins (which every
//!   agent is told to follow).
//! - The transport is a trait. The first one keeps batches on git refs,
//!   `refs/ravnori/<project>/<author>`: of the project's own remote, or (`--via`) of a
//!   separate memory repository that can hold many projects. That repository's access
//!   rules decide who reads them. A database table can replace it without changing what
//!   is sent or applied.

use crate::{forget, text};
use anyhow::{Context, Result, bail, ensure};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Memories per batch (one commit on the git transport).
pub const BATCH: usize = 500;
/// The meta key holding every synced project's state.
const META: &str = "sync.projects";
/// Where a stored memory's origin says it came from a teammate.
const TEAM: &str = "team:";

/// One memory as it travels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub origin: String,
    pub origin_id: String,
    pub kind: String,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub title: Option<String>,
    pub subtitle: Option<String>,
    pub narrative: Option<String>,
    pub facts: Option<String>,
    pub concepts: Option<String>,
    pub files_read: Option<String>,
    pub files_modified: Option<String>,
    pub data: Option<String>,
    pub model: Option<String>,
    pub created_at: Option<i64>,
}

/// A batch from a teammate: whose, where their stream is read up to after it, its bytes.
pub struct Pulled {
    pub author: String,
    pub cursor: String,
    pub bytes: Vec<u8>,
}

/// Where batches are kept.
pub trait Transport {
    /// Store one batch at the end of `author`'s stream, durably. Nothing is published yet.
    fn append(&self, author: &str, batch: &[u8]) -> Result<()>;
    /// Make `author`'s appended batches visible to the team.
    fn publish(&self, author: &str) -> Result<()>;
    /// Every other author's batches after `after` (author -> cursor), oldest first.
    fn pull(&self, me: &str, after: &BTreeMap<String, String>) -> Result<Vec<Pulled>>;
}

/// A synced project's state on this machine.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct State {
    /// The checkout its git commands run in.
    pub root: PathBuf,
    /// This machine's name in the team: `<email name>.<host>`.
    pub author: String,
    /// The highest memory id here already appended.
    pub pushed: i64,
    /// Per teammate: where their stream has been read up to.
    pub pulled: BTreeMap<String, String>,
    /// The memory repository's URL when not the project's own `origin`.
    #[serde(default)]
    pub via: Option<String>,
}

/// What one sync did for one project.
#[derive(Debug, Default)]
pub struct Report {
    pub project: String,
    pub sent: usize,
    /// None when published, else why not (the batches go out next time).
    pub publish_error: Option<String>,
    /// Memories added per teammate.
    pub received: BTreeMap<String, usize>,
    /// Batches whose content could not be read (skipped, so the rest still arrive).
    pub unreadable: usize,
    pub pull_error: Option<String>,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: sent {}", self.project, self.sent)?;
        if let Some(e) = &self.publish_error {
            write!(f, " (kept here, not pushed: {e})")?;
        }
        let got: usize = self.received.values().sum();
        write!(f, ", received {got}")?;
        for (a, n) in self.received.iter().filter(|(_, n)| **n > 0) {
            write!(f, " [{a}: {n}]")?;
        }
        if self.unreadable > 0 {
            write!(f, " ({} unreadable batches skipped)", self.unreadable)?;
        }
        if let Some(e) = &self.pull_error {
            write!(f, " (could not fetch: {e})")?;
        }
        Ok(())
    }
}

/// The teammate who wrote a stored memory, from its origin.
pub fn author_of(origin: &str) -> Option<&str> {
    origin.strip_prefix(TEAM)
}

/// " · from <teammate>" for a memory synced from a teammate, else empty: an agent must
/// be able to tell a teammate's note from its own user's.
pub fn from_suffix(origin: &str) -> String {
    author_of(origin)
        .map(|a| format!(" · from {a}"))
        .unwrap_or_default()
}

/// `from_suffix` for memory `id`.
pub fn from_label(conn: &Connection, id: i64) -> String {
    conn.prepare_cached("SELECT origin FROM memories WHERE id = ?1")
        .and_then(|mut st| st.query_row([id], |r| r.get::<_, String>(0)))
        .map(|o| from_suffix(&o))
        .unwrap_or_default()
}

pub fn projects(conn: &Connection) -> Result<BTreeMap<String, State>> {
    let v: Option<String> = conn
        .query_row("SELECT v FROM meta WHERE k = ?1", [META], |r| r.get(0))
        .optional()?;
    Ok(match v {
        Some(s) => serde_json::from_str(&s).context("read the sync settings")?,
        None => BTreeMap::new(),
    })
}

fn save(conn: &Connection, all: &BTreeMap<String, State>) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(k, v) VALUES (?1, ?2) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        params![META, serde_json::to_string(all)?],
    )?;
    Ok(())
}

/// Share the project checked out at `cwd`, through its own `origin` or through the
/// memory repository `via`: its memories from now on, or all of them with `history`.
/// The remote is reached once here, so a wrong URL or missing access shows at once.
/// Returns (project, state).
pub fn enable(
    conn: &Connection,
    cwd: &Path,
    history: bool,
    via: Option<&str>,
) -> Result<(String, State)> {
    let root = crate::files::repo_root(cwd).context("not inside a git checkout")?;
    let project = crate::project::Resolver::default()
        .resolve(Some(&cwd.to_string_lossy()), None)
        .context("cannot resolve this project")?;
    let probe = State {
        root: root.clone(),
        via: via.map(str::to_string),
        ..State::default()
    };
    let git = transport(&project, &probe)?;
    git.run(&["ls-remote", "--quiet", git.remote, "refs/ravnori/x"])
        .with_context(|| match via {
            Some(url) => format!("cannot reach the memory repository {url}"),
            None => "cannot reach this checkout's `origin` remote to sync through".into(),
        })?;
    let mut all = projects(conn)?;
    let pushed = if history {
        0
    } else {
        conn.query_row("SELECT coalesce(max(id), 0) FROM memories", [], |r| {
            r.get(0)
        })?
    };
    let state = all.entry(project.clone()).or_default();
    state.root = root.clone();
    state.author = author(&root, &project);
    state.via = via.map(str::to_string);
    // Turning it on again keeps what was already sent, unless history asks for more.
    if history || state.pulled.is_empty() && state.pushed == 0 {
        state.pushed = pushed;
    }
    let state = state.clone();
    save(conn, &all)?;
    Ok((project, state))
}

/// Stop sharing `project`. Memories already received stay.
pub fn disable(conn: &Connection, project: &str) -> Result<bool> {
    let mut all = projects(conn)?;
    let had = all.remove(project).is_some();
    save(conn, &all)?;
    Ok(had)
}

/// Sync every shared project through its git remote.
pub fn run(conn: &mut Connection) -> Result<Vec<Report>> {
    let all = projects(conn)?;
    let mut out = Vec::new();
    for (project, state) in all {
        let git = transport(&project, &state)?;
        out.push(sync_project(conn, &project, &git)?);
    }
    Ok(out)
}

/// Sync one shared project through its own transport.
pub fn sync_one(conn: &mut Connection, project: &str) -> Result<Report> {
    let state = projects(conn)?
        .remove(project)
        .with_context(|| format!("{project} is not shared (run `rvn sync on` in its checkout)"))?;
    let git = transport(project, &state)?;
    sync_project(conn, project, &git)
}

/// The git transport for a shared project: its checkout's `origin`, or a bare local
/// repository under the data folder whose `origin` is the memory repository.
pub fn transport(project: &str, state: &State) -> Result<GitRefs> {
    let key = space(project);
    let Some(url) = &state.via else {
        return Ok(GitRefs::new(&state.root, &key));
    };
    // One local copy per memory repository and person (a test runs two on one disk).
    let dir = crate::db::data_dir().join("sync").join(format!(
        "{}@{}.git",
        slug_path(url).replace('/', "_"),
        person(&state.author)
    ));
    if !dir.join("HEAD").exists() {
        std::fs::create_dir_all(&dir)?;
        GitRefs::new(&dir, &key).run(&["init", "--quiet", "--bare"])?;
    }
    let git = GitRefs::new(&dir, &key);
    if git.run(&["remote", "get-url", "origin"]).ok().as_deref() != Some(url.as_str()) {
        let _ = git.run(&["remote", "remove", "origin"]);
        git.run(&["remote", "add", "origin", url])?;
    }
    Ok(git)
}

/// A project's namespace in a memory repository: its remote without the checkout tag,
/// so teammates whose folders are named differently still meet in one place, closed by
/// a segment `slug_path` never produces: `a/b` must not read `a/b/c`'s streams, since
/// git's `*` matches across `/`.
pub fn space(project: &str) -> String {
    let s = slug_path(project.split('#').next().unwrap_or(project));
    format!("{}/{SPACE_END}", if s.is_empty() { "project" } else { &s })
}

/// Ends a project's namespace in ref names: `=` is valid in a ref, never in a slug.
const SPACE_END: &str = "=";

/// A path safe as part of a git ref: each segment slugged ('.' becomes '_', so no `..`,
/// leading dot or `.lock`), empty segments dropped.
fn slug_path(s: &str) -> String {
    s.split(['/', ':'])
        .map(|seg| slug(&seg.replace('.', "_")))
        .filter(|seg| !seg.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

/// Add the teammates' new memories for `project`, then send this machine's new ones.
pub fn sync_project(conn: &mut Connection, project: &str, t: &dyn Transport) -> Result<Report> {
    let mut all = projects(conn)?;
    let Some(mut state) = all.get(project).cloned() else {
        bail!("{project} is not shared (run `rvn sync on` in its checkout)");
    };
    let mut report = Report {
        project: project.to_string(),
        ..Report::default()
    };
    // Fetch first: the transport then knows where this author's stream stands remotely.
    match t.pull(&state.author, &state.pulled) {
        Ok(batches) => {
            for b in batches {
                // One malformed batch from one teammate must not stop everyone else's.
                match decode(&b.bytes) {
                    Ok(records) => {
                        let n = apply(conn, project, &b.author, &records)?;
                        *report
                            .received
                            .entry(person(&b.author).to_string())
                            .or_default() += n;
                    }
                    Err(_) => report.unreadable += 1,
                }
                state.pulled.insert(b.author, b.cursor);
                all.insert(project.to_string(), state.clone());
                save(conn, &all)?;
            }
        }
        Err(e) => report.pull_error = Some(format!("{e:#}")),
    }
    loop {
        let rows = outbox(conn, project, state.pushed, BATCH)?;
        let Some(last) = rows.last().map(|(id, _)| *id) else {
            break;
        };
        let records: Vec<Record> = rows.into_iter().map(|(_, r)| r).collect();
        t.append(&state.author, &encode(&records)?)?;
        report.sent += records.len();
        state.pushed = last;
        all.insert(project.to_string(), state.clone());
        save(conn, &all)?;
    }
    if let Err(e) = t.publish(&state.author) {
        report.publish_error = Some(format!("{e:#}"));
    }
    Ok(report)
}

/// This machine's memories for `project` after id `after`, ready to send.
fn outbox(
    conn: &Connection,
    project: &str,
    after: i64,
    limit: usize,
) -> Result<Vec<(i64, Record)>> {
    let mut st = conn.prepare(&format!(
        "SELECT id, origin, origin_id, kind, type, title, subtitle, narrative, facts, concepts,
                files_read, files_modified, data, model, created_at
         FROM memories m
         WHERE project = ?1 AND id > ?2 AND origin IN ('ravnori', 'claude-mem')
           AND kind IN ('observation', 'summary') AND coalesce(type, '') != 'sensitive' AND {}
         ORDER BY id LIMIT ?3",
        crate::scripted::MEMORY_NOT_SCRIPTED
    ))?;
    let red = |s: Option<String>| s.map(|s| text::redact(&s));
    let rows = st.query_map(params![project, after, limit as i64], |r| {
        Ok((
            r.get(0)?,
            Record {
                origin: r.get(1)?,
                origin_id: r.get(2)?,
                kind: r.get(3)?,
                type_: r.get(4)?,
                title: red(r.get(5)?),
                subtitle: red(r.get(6)?),
                narrative: red(r.get(7)?),
                facts: red(r.get(8)?),
                concepts: r.get(9)?,
                files_read: r.get(10)?,
                files_modified: r.get(11)?,
                data: red(r.get(12)?),
                model: r.get(13)?,
                created_at: r.get(14)?,
            },
        ))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Add a teammate's records to `project` here. Returns how many were new.
fn apply(conn: &mut Connection, project: &str, author: &str, records: &[Record]) -> Result<usize> {
    let tx = conn.transaction()?;
    let project_forgotten = tx
        .query_row(
            "SELECT 1 FROM forgotten WHERE kind = 'project' AND key = ?1",
            [project],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    let origin = format!("{TEAM}{}", person(author));
    let mut added = 0;
    {
        let mut ins = tx.prepare(
            "INSERT OR IGNORE INTO memories(session_id, project, kind, type, title, subtitle, narrative, facts,
                 concepts, files_read, files_modified, data, origin, origin_id, model, created_at)
             VALUES (NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;
        for r in records {
            // Pins and anything sensitive are never accepted, whatever a batch says.
            let accepted = matches!(r.kind.as_str(), "observation" | "summary")
                && r.type_.as_deref() != Some("sensitive");
            let origin_id = format!("{}:{}", r.origin, r.origin_id);
            if !accepted || project_forgotten || forget::memory_forgotten(&tx, &origin, &origin_id)?
            {
                continue;
            }
            added += ins.execute(params![
                project,
                r.kind,
                r.type_,
                r.title,
                r.subtitle,
                r.narrative,
                r.facts,
                r.concepts,
                r.files_read,
                r.files_modified,
                r.data,
                origin,
                origin_id,
                r.model,
                r.created_at,
            ])?;
        }
    }
    tx.commit()?;
    Ok(added)
}

/// Records as gzip'd JSON lines.
pub fn encode(records: &[Record]) -> Result<Vec<u8>> {
    let mut gz = GzEncoder::new(Vec::new(), Compression::best());
    for r in records {
        serde_json::to_writer(&mut gz, r)?;
        gz.write_all(b"\n")?;
    }
    Ok(gz.finish()?)
}

pub fn decode(bytes: &[u8]) -> Result<Vec<Record>> {
    let mut s = String::new();
    GzDecoder::new(bytes)
        .read_to_string(&mut s)
        .context("unpack a sync batch")?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("read a sync record"))
        .collect()
}

/// The checkout part of a project id (`github.com/o/r#second` -> `second`), else `main`:
/// two checkouts of one remote on one machine are separate projects, so separate streams.
fn checkout_of(project: &str) -> String {
    let tag = slug(project.rsplit_once('#').map(|(_, t)| t).unwrap_or(""));
    if tag.is_empty() { "main".into() } else { tag }
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| c.to_ascii_lowercase())
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(40)
        .collect()
}

/// `<email name>.<host>/<checkout>`, in the characters a git ref name allows: one stream
/// per person, machine and checkout. A teammate is everything before the `/`.
fn author(root: &Path, project: &str) -> String {
    let email = GitRefs::new(root, "")
        .run(&["config", "user.email"])
        .unwrap_or_default();
    let name = email.split('@').next().unwrap_or("").to_string();
    let (n, h) = (slug(&name), slug(&crate::backup::hostname()));
    format!(
        "{}.{}/{}",
        if n.is_empty() { "unknown" } else { &n },
        if h.is_empty() { "host" } else { &h },
        checkout_of(project)
    )
}

/// The person and machine of a stream (`fery.box/main` -> `fery.box`).
fn person(stream: &str) -> &str {
    stream.split_once('/').map(|(p, _)| p).unwrap_or(stream)
}

/// Batches on git refs of a remote, under one project's namespace: one ref per author,
/// one commit per batch, holding one file. A plain clone fetches none of them.
pub struct GitRefs {
    root: PathBuf,
    remote: &'static str,
    /// The project's namespace (`space`).
    key: String,
}

const FILE: &str = "batch.jsonl.gz";

impl GitRefs {
    /// Git commands run in `root` (a checkout, or the bare memory repository); refs live
    /// under `refs/ravnori/<key>/`.
    pub fn new(root: &Path, key: &str) -> GitRefs {
        GitRefs {
            root: root.to_path_buf(),
            remote: "origin",
            key: key.to_string(),
        }
    }

    fn own(&self, author: &str) -> String {
        format!("refs/ravnori/{}/{author}", self.key)
    }

    /// Where fetched teammates' refs are kept here.
    fn team(&self) -> String {
        format!("refs/ravnori-team/{}/", self.key)
    }

    fn bytes(&self, args: &[&str], input: Option<&[u8]>) -> Result<Vec<u8>> {
        let mut child = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_AUTHOR_NAME", "ravnori")
            .env("GIT_AUTHOR_EMAIL", "sync@ravnori")
            .env("GIT_COMMITTER_NAME", "ravnori")
            .env("GIT_COMMITTER_EMAIL", "sync@ravnori")
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("run git")?;
        if let (Some(data), Some(mut stdin)) = (input, child.stdin.take()) {
            stdin.write_all(data)?;
        }
        let out = child.wait_with_output()?;
        ensure!(
            out.status.success(),
            "git {}: {}",
            args.first().copied().unwrap_or(""),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        Ok(out.stdout)
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.bytes(args, None)?)
            .trim()
            .to_string())
    }

    fn input(&self, args: &[&str], input: &[u8]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.bytes(args, Some(input))?)
            .trim()
            .to_string())
    }

    fn commit_exists(&self, rev: &str) -> bool {
        self.run(&["cat-file", "-e", &format!("{rev}^{{commit}}")])
            .is_ok()
    }
}

impl Transport for GitRefs {
    fn append(&self, author: &str, batch: &[u8]) -> Result<()> {
        let own = self.own(author);
        let blob = self.input(&["hash-object", "-w", "--stdin"], batch)?;
        let tree = self.input(
            &["mktree"],
            format!("100644 blob {blob}\t{FILE}\n").as_bytes(),
        )?;
        // A machine set up again continues its stream as last fetched from the remote.
        let parent = self
            .run(&["rev-parse", "--verify", "--quiet", &own])
            .or_else(|_| {
                self.run(&[
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    &format!("{}{author}", self.team()),
                ])
            })
            .ok();
        let mut args = vec!["commit-tree", tree.as_str(), "-m", "ravnori memory batch"];
        if let Some(p) = &parent {
            args.extend(["-p", p.as_str()]);
        }
        let commit = self.run(&args)?;
        // Moves only from the parent read above: two syncs at once cannot drop a batch.
        let old = parent.unwrap_or_else(|| "0".repeat(40));
        self.run(&["update-ref", &own, &commit, &old])?;
        Ok(())
    }

    fn publish(&self, author: &str) -> Result<()> {
        let own = self.own(author);
        if !self.commit_exists(&own) {
            return Ok(());
        }
        self.run(&["push", "--quiet", self.remote, &format!("{own}:{own}")])?;
        Ok(())
    }

    fn pull(&self, me: &str, after: &BTreeMap<String, String>) -> Result<Vec<Pulled>> {
        // Only this project's namespace: a memory repository holds other projects too.
        let team = self.team();
        self.run(&[
            "fetch",
            "--quiet",
            "--no-tags",
            self.remote,
            &format!("+refs/ravnori/{}/*:{team}*", self.key),
        ])?;
        let refs = self.run(&["for-each-ref", "--format=%(refname)", &team])?;
        let mut out = Vec::new();
        for r in refs.lines() {
            let Some(author) = r.strip_prefix(team.as_str()) else {
                continue;
            };
            // This machine's other checkouts are its own projects, not teammates.
            if author.is_empty() || person(author) == person(me) {
                continue;
            }
            let mut args = vec!["rev-list", "--reverse", r];
            let not;
            if let Some(c) = after.get(author).filter(|c| self.commit_exists(c)) {
                not = format!("^{c}");
                args.push(&not);
            }
            for commit in self.run(&args)?.lines().filter(|l| !l.is_empty()) {
                // A commit without a readable batch (pushed by hand, another tool) comes
                // back empty: it is counted as unreadable and passed, never blocking the rest.
                let bytes = self
                    .bytes(&["cat-file", "blob", &format!("{commit}:{FILE}")], None)
                    .unwrap_or_default();
                out.push(Pulled {
                    author: author.to_string(),
                    cursor: commit.to_string(),
                    bytes,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, kind: &str, ty: Option<&str>) -> Record {
        Record {
            origin: "ravnori".into(),
            origin_id: id.into(),
            kind: kind.into(),
            type_: ty.map(Into::into),
            title: Some(format!("title {id}")),
            subtitle: None,
            narrative: Some("n".into()),
            facts: Some("[]".into()),
            concepts: None,
            files_read: None,
            files_modified: Some("[\"src/a.rs\"]".into()),
            data: None,
            model: None,
            created_at: Some(1),
        }
    }

    #[test]
    fn projects_get_closed_namespaces_safe_in_ref_names() {
        assert_eq!(
            space("gitlab.solveeducation.org/solveearn/solveeducation#second"),
            "gitlab_solveeducation_org/solveearn/solveeducation/="
        );
        // One project's namespace is never the start of another's.
        assert!(!space("h/a/b/c").starts_with(&space("h/a/b")));
        // No `..`, leading dot, `.lock` or empty segment reaches git.
        assert_eq!(space("../x.lock//y"), "__/x_lock/y/=");
        assert_eq!(space("/home/ana/proj"), "home/ana/proj/=");
        assert_eq!(space(""), "project/=");
    }

    #[test]
    fn batches_round_trip() {
        let v = vec![
            rec("a", "observation", Some("bugfix")),
            rec("b", "summary", None),
        ];
        assert_eq!(decode(&encode(&v).unwrap()).unwrap(), v);
    }

    #[test]
    fn pins_and_sensitive_records_are_refused() {
        let dir = crate::TempDir::new("sync");
        let mut c = crate::db::open(&dir.join("t.db")).unwrap();
        let n = apply(
            &mut c,
            "p",
            "ana.box",
            &[
                rec("1", "observation", Some("bugfix")),
                rec("2", "pinned", None),
                rec("3", "observation", Some("sensitive")),
            ],
        )
        .unwrap();
        assert_eq!(n, 1);
        let origin: String = c
            .query_row("SELECT origin FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(author_of(&origin), Some("ana.box"));
    }
}
