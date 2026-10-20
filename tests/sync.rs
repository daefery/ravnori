//! Team memory through git refs: two machines share one project through a bare remote.
//! Each one's distilled memories reach the other, labelled with their author; nothing
//! private (pins, sensitive, transcripts) leaves; syncing again adds nothing; a memory
//! one side forgot stays forgotten; and a sync that cannot reach the remote keeps its
//! batches and sends them next time. Its own test binary: it sets RAVNORI_HOME.

use ravnori::sync;
use rusqlite::{Connection, params};
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .unwrap();
    assert!(
        ok.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&ok.stderr)
    );
}

/// A checkout of `remote` at `dir` whose user is `email`.
fn checkout(remote: &Path, dir: &Path, email: &str) {
    git(
        dir.parent().unwrap(),
        &[
            "clone",
            "--quiet",
            &remote.to_string_lossy(),
            &dir.to_string_lossy(),
        ],
    );
    git(dir, &["config", "user.email", email]);
}

fn memory(c: &Connection, project: &str, origin_id: &str, kind: &str, ty: &str, title: &str) {
    c.execute(
        "INSERT INTO memories(project, kind, type, title, narrative, origin, origin_id, created_at)
         VALUES (?1, ?2, ?3, ?4, 'n', 'ravnori', ?5, 1000)",
        params![project, kind, ty, title, origin_id],
    )
    .unwrap();
}

fn titles(c: &Connection, project: &str) -> Vec<(String, String)> {
    let mut st = c
        .prepare("SELECT origin, title FROM memories WHERE project = ?1 AND origin LIKE 'team:%' ORDER BY title")
        .unwrap();
    st.query_map([project], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn two_machines_share_one_project_through_git_refs() {
    let home = ravnori::TempDir::new("sync");
    // SAFETY: set before any other thread starts.
    unsafe { std::env::set_var("RAVNORI_HOME", &home) };

    // One project on a shared remote with one commit, cloned on two machines.
    let remote = home.join("remote.git");
    std::fs::create_dir_all(&remote).unwrap();
    git(&remote, &["init", "--quiet", "--bare"]);
    let seed = home.join("seed");
    checkout(&remote, &seed, "t@t");
    std::fs::write(seed.join("README"), "x").unwrap();
    git(&seed, &["add", "."]);
    git(&seed, &["commit", "--quiet", "-m", "init"]);
    git(&seed, &["push", "--quiet", "origin", "HEAD"]);
    let (fery_dir, ana_dir) = (home.join("shop-fery"), home.join("shop-ana"));
    checkout(&remote, &fery_dir, "fery@example.com");
    checkout(&remote, &ana_dir, "ana@example.com");

    let mut fery = ravnori::db::open(&home.join("fery.db")).unwrap();
    let mut ana = ravnori::db::open(&home.join("ana.db")).unwrap();
    // Their folders are named differently, so each machine has its own project id for
    // the same remote: memories land in the receiver's own project.
    let (p, fs) = sync::enable(&fery, &fery_dir, true, None).unwrap();
    let (pa, as_) = sync::enable(&ana, &ana_dir, true, None).unwrap();
    assert_ne!(p, pa);
    let fery_name = fs.author.split('/').next().unwrap().to_string();
    assert!(
        fery_name.starts_with("fery.") && as_.author.starts_with("ana."),
        "{fery_name}"
    );

    // Fery: two memories to share, plus a pin, a sensitive note and another project's
    // memory, none of which may leave.
    memory(
        &fery,
        &p,
        "s@1-2#0",
        "observation",
        "bugfix",
        "Fery fixed the retry loop",
    );
    memory(
        &fery,
        &p,
        "s@1-2#1",
        "summary",
        "summary",
        "Fery's session summary",
    );
    memory(
        &fery,
        &p,
        "s@1-2#2",
        "observation",
        "sensitive",
        "Fery's private note",
    );
    memory(
        &fery,
        "github.com/other/repo",
        "s@3-4#0",
        "observation",
        "bugfix",
        "Other project",
    );
    ravnori::forget::remember(&fery, "Fery's pin", Some(&p)).unwrap();
    // Ana: one memory.
    memory(
        &ana,
        &pa,
        "a@1-1#0",
        "observation",
        "decision",
        "Ana chose Postgres",
    );

    // A ref someone pushed by hand, holding no batch, must not block anyone's batches.
    git(
        &seed,
        &[
            "push",
            "--quiet",
            "origin",
            &format!("HEAD:refs/ravnori/{}/someone.box/main", sync::space(&p)),
        ],
    );

    let fr = sync::sync_one(&mut fery, &p).unwrap();
    assert_eq!((fr.sent, fr.publish_error.as_deref()), (2, None));
    assert_eq!((fr.unreadable, fr.pull_error.as_deref()), (1, None), "{fr}");
    let ar = sync::sync_one(&mut ana, &pa).unwrap();
    assert_eq!(ar.sent, 1);
    assert_eq!(ar.received.values().sum::<usize>(), 2);
    let fr = sync::sync_one(&mut fery, &p).unwrap();
    assert_eq!((fr.sent, fr.received.values().sum::<usize>()), (0, 1));

    let ana_has = titles(&ana, &pa);
    assert_eq!(
        ana_has.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>(),
        ["Fery fixed the retry loop", "Fery's session summary"]
    );
    assert_eq!(ana_has[0].0, format!("team:{fery_name}"));
    assert_eq!(titles(&fery, &p)[0].1, "Ana chose Postgres");
    // A teammate's memory is labelled wherever an agent sees it.
    let ctx = ravnori::context::build(
        &ana,
        &ravnori::context::Options {
            project: &pa,
            current: None,
            budget_chars: 8000,
            sessions: 5,
            turns: 3,
            observations: 30,
        },
    )
    .unwrap();
    assert!(ctx.contains(&format!("from {fery_name}")), "{ctx}");

    // Received memories are never sent on: syncing again moves nothing either way.
    for (c, proj) in [(&mut ana, &pa), (&mut fery, &p)] {
        let r = sync::sync_one(c, proj).unwrap();
        assert_eq!((r.sent, r.received.values().sum::<usize>()), (0, 0), "{r}");
    }

    // Ana forgets one of Fery's memories; a fresh read of his stream does not bring it back.
    let id: i64 = ana
        .query_row(
            "SELECT id FROM memories WHERE title = 'Fery fixed the retry loop'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    ravnori::forget::forget(&mut ana, &[id.to_string()], None, None).unwrap();
    ana.execute("UPDATE meta SET v = json_set(v, '$.\"' || ?1 || '\".pulled', json('{}')) WHERE k = 'sync.projects'", [&pa])
        .unwrap();
    sync::sync_one(&mut ana, &pa).unwrap();
    assert_eq!(titles(&ana, &pa).len(), 1);

    // The remote is unreachable: Fery's new memory is kept as a batch and goes next time.
    memory(
        &fery,
        &p,
        "s@5-6#0",
        "observation",
        "feature",
        "Fery added jitter",
    );
    std::fs::rename(&remote, home.join("away.git")).unwrap();
    let r = sync::sync_one(&mut fery, &p).unwrap();
    assert_eq!(r.sent, 1);
    assert!(r.publish_error.is_some() && r.pull_error.is_some(), "{r}");
    std::fs::rename(home.join("away.git"), &remote).unwrap();
    let r = sync::sync_one(&mut fery, &p).unwrap();
    assert_eq!((r.sent, r.publish_error.as_deref()), (0, None), "{r}");
    sync::sync_one(&mut ana, &pa).unwrap();
    assert!(
        titles(&ana, &pa)
            .iter()
            .any(|(_, t)| t == "Fery added jitter")
    );

    // A plain clone of the repository fetches none of it.
    let plain = home.join("plain");
    checkout(&remote, &plain, "x@x");
    let out = Command::new("git")
        .arg("-C")
        .arg(&plain)
        .args(["for-each-ref", "refs/ravnori"])
        .output()
        .unwrap();
    assert!(out.stdout.is_empty());
}

/// Two projects kept in one separate memory repository (`--via`), the second one's name
/// starting with the first one's: each machine receives only the project it syncs, and
/// the code repositories get no refs at all.
#[test]
fn one_memory_repository_holds_many_projects_apart() {
    let home = ravnori::TempDir::new("sync-via");
    // SAFETY: set before any other thread starts.
    unsafe { std::env::set_var("RAVNORI_HOME", &home) };
    let bare = |name: &str| {
        let d = home.join(name);
        std::fs::create_dir_all(&d).unwrap();
        git(&d, &["init", "--quiet", "--bare"]);
        d
    };
    let memory_repo = bare("memory.git");
    let url = memory_repo.to_string_lossy().into_owned();
    let (shop, shop_admin) = (bare("shop.git"), bare("shop/admin.git"));
    let open = |who: &str, code: &Path, email: &str| {
        let dir = home.join(who);
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join(code.file_stem().unwrap());
        checkout(code, &co, email);
        co
    };
    let fery_shop = open("fery", &shop, "fery@example.com");
    let fery_admin = open("fery", &shop_admin, "fery@example.com");
    let ana_shop = open("ana", &shop, "ana@example.com");
    let ana_admin = open("ana", &shop_admin, "ana@example.com");

    let mut fery = ravnori::db::open(&home.join("fery.db")).unwrap();
    let mut ana = ravnori::db::open(&home.join("ana.db")).unwrap();
    let (fs, _) = sync::enable(&fery, &fery_shop, true, Some(&url)).unwrap();
    let (fa, _) = sync::enable(&fery, &fery_admin, true, Some(&url)).unwrap();
    let (as_, _) = sync::enable(&ana, &ana_shop, true, Some(&url)).unwrap();
    assert!(fa.starts_with(&fs), "{fa} {fs}");
    memory(&fery, &fs, "s@1-1#0", "observation", "bugfix", "Shop fix");
    memory(&fery, &fa, "s@2-2#0", "observation", "bugfix", "Admin fix");
    for r in sync::run(&mut fery).unwrap() {
        assert_eq!((r.sent, r.publish_error.as_deref()), (1, None), "{r}");
    }

    // Ana syncs only the shop: she gets the shop memory and nothing of the admin.
    let r = sync::run(&mut ana).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].received.values().sum::<usize>(), 1, "{}", r[0]);
    assert_eq!(titles(&ana, &as_)[0].1, "Shop fix");
    // When she shares the admin too, its memory arrives there and only there.
    let (aa, _) = sync::enable(&ana, &ana_admin, true, Some(&url)).unwrap();
    sync::run(&mut ana).unwrap();
    assert_eq!(
        titles(&ana, &aa)
            .iter()
            .map(|t| t.1.as_str())
            .collect::<Vec<_>>(),
        ["Admin fix"]
    );
    assert_eq!(titles(&ana, &as_).len(), 1);

    // The memories are in the memory repository; the code repositories have no refs.
    let refs = |d: &Path| {
        let o = Command::new("git")
            .arg("-C")
            .arg(d)
            .args(["for-each-ref", "refs/ravnori"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&o.stdout).lines().count()
    };
    assert_eq!(refs(&memory_repo), 2);
    assert_eq!((refs(&shop), refs(&shop_admin)), (0, 0));

    // A wrong URL is refused when sharing starts, not on the first sync.
    let bad = home.join("missing.git").to_string_lossy().into_owned();
    let mut other = ravnori::db::open(&home.join("other.db")).unwrap();
    assert!(sync::enable(&other, &fery_shop, true, Some(&bad)).is_err());
    assert!(sync::run(&mut other).unwrap().is_empty());
}
