import glob
import hashlib
import os
import tempfile
import threading

import astra.memory as ms
from astra.memory import (
    CommandStore,
    MemoryNotFoundError,
    TripleMemoryStore,
    atomic_save,
)


def dir_hashes(directory: str) -> dict:
    out = {}
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                out[os.path.relpath(path, directory)] = hashlib.sha256(f.read()).hexdigest()
    return out


def expect_error(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} from {fn.__name__}")


# ---------------------------------------------------------------------
def test_unique_ids_and_validation(tmp):
    store = TripleMemoryStore(tmp)
    ids = [store.add_memory("roum", f"fact number {i}", "explicit_fact", "test") for i in range(200)]
    assert len(set(ids)) == 200, "Ids must be unique even within the same millisecond."

    expect_error(ValueError, store.add_memory, "nobody", "x", "explicit_fact", "t")
    expect_error(ValueError, store.add_memory, "roum", "x", "not_a_type", "t")
    expect_error(ValueError, store.add_memory, "roum", "   ", "explicit_fact", "t")
    expect_error(ValueError, store.add_memory, "roum", "x", "explicit_fact", "t", id="mem_custom")
    expect_error(ValueError, store.add_memory, "roum", "x", "explicit_fact", "t", timestamp="nope")
    expect_error(ValueError, store.get_active_memories, "nobody")

    old_id = store.add_memory("roum", "backdated", "explicit_fact", "t",
                              timestamp="2020-01-01T00:00:00+00:00")
    assert store.get_memory("roum", old_id)["timestamp"].startswith("2020-01-01")
    print("✓ unique ids, validation, and backdating")


def test_dedupe_and_reinforcement(tmp):
    store = TripleMemoryStore(tmp)
    a = store.add_memory("roum", "Roum likes tea", "explicit_preference", "t", confidence=0.6, tags=["drinks"])
    b = store.add_memory("roum", "  roum LIKES   tea ", "explicit_preference", "t", tags=["tea"])
    assert a == b and len(store.get_active_memories("roum")) == 1
    mem = store.get_memory("roum", a)
    assert mem["reinforcement_count"] == 2
    assert abs(mem["confidence"] - 0.65) < 1e-9
    assert set(mem["tags"]) == {"drinks", "tea"}

    c = store.add_memory("roum", "Roum likes tea", "explicit_preference", "t", dedupe=False)
    assert c != a and len(store.get_active_memories("roum")) == 2
    d = store.add_memory("roum", "Roum likes tea", "explicit_fact", "t")
    assert d not in (a, c), "Same text with a different type is not a duplicate."
    print("✓ dedupe reinforces instead of duplicating")


def test_lifecycle(tmp):
    store = TripleMemoryStore(tmp)
    mid = store.add_memory("self", "Astra is shy", "self_belief", "t")

    updated = store.update_memory("self", mid, content="Astra is a little shy", confidence=0.4, tags=["trait"])
    assert updated["content"] == "Astra is a little shy" and updated["confidence"] == 0.4
    assert "updated_at" in updated
    expect_error(ValueError, store.update_memory, "self", mid)
    expect_error(MemoryNotFoundError, store.update_memory, "self", "mem_missing", content="x")

    store.archive_memory("self", mid, reason="outdated")
    assert store.get_active_memories("self") == []
    assert store.get_memory("self", mid)["archive_reason"] == "outdated"
    store.restore_memory("self", mid)
    assert len(store.get_active_memories("self")) == 1

    new_id = store.supersede_memory("self", mid, "Astra is confident now")
    assert store.get_memory("self", mid)["status"] == "superseded"
    assert store.get_memory("self", mid)["superseded_by"] == new_id
    assert store.get_memory("self", new_id)["supersedes"] == mid
    assert store.get_memory("self", new_id)["type"] == "self_belief"
    assert [m["id"] for m in store.get_active_memories("self")] == [new_id]
    expect_error(ValueError, store.supersede_memory, "self", mid, "again")
    expect_error(ValueError, store.restore_memory, "self", mid)

    reopened = TripleMemoryStore(tmp)
    assert [m["id"] for m in reopened.get_active_memories("self")] == [new_id]
    print("✓ update / archive / restore / supersede persist correctly")


def test_returns_copies_and_stats(tmp):
    store = TripleMemoryStore(tmp)
    store.add_memory("roum", "fact one", "explicit_fact", "t", tags=["a"])
    store.add_memory("roum", "pref one", "explicit_preference", "t")
    got = store.get_active_memories("roum")
    got[0]["content"] = "HACKED"
    got[0]["tags"].append("hacked")
    assert store.get_active_memories("roum")[0]["content"] == "fact one"
    assert store.get_active_memories("roum")[0]["tags"] == ["a"]

    stats = store.stats()
    assert stats["roum"]["active"] == 2 and stats["roum"]["by_type"]["explicit_fact"] == 1
    assert stats["self"]["total"] == 0 and stats["journal_entries"] == 0
    print("✓ reads return copies; stats are correct")


def test_reads_never_write(tmp):
    store = TripleMemoryStore(tmp)
    store.add_memory("roum", "a fact", "explicit_fact", "t")
    store.add_journal_entry("title", "obs", "conv")
    before = dir_hashes(tmp)

    reopened = TripleMemoryStore(tmp)
    reopened.get_active_memories("roum")
    reopened.get_memories("roum", status=None)
    reopened.stats()
    reopened.get_journal()
    CommandStore(tmp).get_last_commands_context()
    assert dir_hashes(tmp) == before
    print("✓ loading and reading leave every file byte-for-byte identical")


def test_rollback_on_save_failure(tmp):
    store = TripleMemoryStore(tmp)
    keep = store.add_memory("roum", "safe fact", "explicit_fact", "t")
    before = dir_hashes(tmp)

    original = ms.atomic_save

    def boom(*args, **kwargs):
        raise OSError("disk full")

    ms.atomic_save = boom
    try:
        expect_error(OSError, store.add_memory, "roum", "doomed", "explicit_fact", "t")
        expect_error(OSError, store.archive_memory, "roum", keep)
        expect_error(OSError, store.update_memory, "roum", keep, content="changed")
        expect_error(OSError, store.add_journal_entry, "t", "o", "c")
    finally:
        ms.atomic_save = original

    active = store.get_active_memories("roum")
    assert [m["content"] for m in active] == ["safe fact"], "RAM must roll back to match disk."
    assert store.get_journal() == []
    assert dir_hashes(tmp) == before
    print("✓ failed saves roll back in-memory state")


def test_corruption_recovery(tmp):
    store = TripleMemoryStore(tmp)
    store.add_memory("roum", "first", "explicit_fact", "t")
    store.add_memory("roum", "second", "explicit_fact", "t")  # creates .bak holding only 'first'
    main = os.path.join(tmp, "roum_model.json")
    assert os.path.exists(main + ".bak")

    with open(main, "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    recovered = TripleMemoryStore(tmp)
    assert [m["content"] for m in recovered.get_active_memories("roum")] == ["first"]
    assert glob.glob(main + ".corrupt-*"), "Corrupt file should be kept aside, not deleted."

    # corrupt with no backup at all -> empty, no crash
    other = tempfile.mkdtemp(dir=tmp)
    with open(os.path.join(other, "self_model.json"), "w", encoding="utf-8") as f:
        f.write("garbage")
    assert TripleMemoryStore(other).get_active_memories("self") == []

    # wrong JSON shape is treated as corruption too
    shape = tempfile.mkdtemp(dir=tmp)
    with open(os.path.join(shape, "roum_model.json"), "w", encoding="utf-8") as f:
        f.write('{"not": "a list"}')
    assert TripleMemoryStore(shape).get_active_memories("roum") == []
    print("✓ corrupt files are quarantined and the backup is used")


def test_atomic_save_edge_cases(tmp):
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        atomic_save("bare_name.json", {"ok": True})  # no directory component
        assert os.path.exists("bare_name.json")
        assert not glob.glob("*.tmp")
        atomic_save("bare_name.json", {"ok": 2})
        assert os.path.exists("bare_name.json.bak")
    finally:
        os.chdir(cwd)

    class Unserializable:
        pass

    target = os.path.join(tmp, "x.json")
    atomic_save(target, [1])
    expect_error(TypeError, atomic_save, target, [Unserializable()])
    assert not glob.glob(os.path.join(tmp, "*.tmp")), "Temp files must be cleaned up on failure."
    print("✓ atomic_save handles bare filenames and cleans up after failures")


def test_thread_safety(tmp):
    store = TripleMemoryStore(tmp)

    def worker(n):
        for i in range(25):
            store.add_memory("roum", f"thread {n} fact {i}", "explicit_fact", "t")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(store.get_active_memories("roum")) == 200
    assert len(TripleMemoryStore(tmp).get_active_memories("roum")) == 200
    print("✓ concurrent writers don't lose memories")


def test_journal_and_export(tmp):
    store = TripleMemoryStore(tmp)
    ids = [store.add_journal_entry(f"t{i}", f"obs {i}", "conv") for i in range(5)]
    assert len(set(ids)) == 5
    assert [e["title"] for e in store.get_journal(limit=2)] == ["t3", "t4"]
    expect_error(ValueError, store.add_journal_entry, "", "obs", "conv")

    store.add_memory("roum", "exported fact", "explicit_fact", "t")
    out = os.path.join(tmp, "snap", "export.json")
    store.export_snapshot(out)
    import json
    with open(out, encoding="utf-8") as f:
        snap = json.load(f)
    assert snap["memories"]["roum"][0]["content"] == "exported fact" and len(snap["journal"]) == 5
    print("✓ journal and export snapshot")


def test_command_store(tmp):
    cs = CommandStore(tmp)
    expect_error(ValueError, cs.add_command, "raw", "   ", "reply")
    expect_error(ValueError, cs.add_command, "raw", "hi", "  ")

    cs.add_command("raw", "hi", "Hello!", once=False)
    assert cs.check_trigger("this is fine") is None, "'hi' must not match inside 'this'."
    assert cs.check_trigger("oh hi there") == "Hello!"
    assert cs.check_trigger("oh hi there") == "Hello!", "once=False commands stay armed."

    a = cs.add_command("raw", "night", "Sleep well", once=False)
    cs.add_command("raw", "good night", "Good night, Roum.", once=False)
    assert cs.check_trigger("Good   NIGHT!") == "Good night, Roum.", "Most specific trigger wins."
    assert cs.add_command("raw", "night", "Sleep well", once=False) == a, "Identical command is reused."

    cs.add_command("raw", "remind me", "Take a break", once=True)
    assert cs.check_trigger("please remind me") == "Take a break"
    assert cs.check_trigger("please remind me") is None, "once=True fires a single time."
    assert all(c["trigger"] != "remind me" for c in CommandStore(tmp).list_commands()), \
        "Consumption must persist across restarts."

    cid = cs.list_commands()[0]["id"]
    assert cs.remove_command(cid) is True and cs.remove_command(cid) is False
    assert len(cs.get_last_commands_context()) == 3 and len(cs.get_last_commands_context(1)) == 1
    print("✓ command store: word-boundary match, specificity, once, persistence, removal")


def test_command_history_cap_and_legacy(tmp):
    original = ms.MAX_COMMAND_HISTORY
    ms.MAX_COMMAND_HISTORY = 5
    try:
        cs = CommandStore(tmp)
        for i in range(8):
            cs.add_command(f"raw {i}", f"trigger{i}", f"resp {i}")
        assert len(cs.history) == 5 and cs.history[-1]["user_input"] == "raw 7"
    finally:
        ms.MAX_COMMAND_HISTORY = original

    legacy_dir = tempfile.mkdtemp(dir=tmp)
    with open(os.path.join(legacy_dir, "active_commands.json"), "w", encoding="utf-8") as f:
        f.write('[{"trigger": "ping", "response": "pong", "once": true, "timestamp": "2024-01-01T00:00:00+00:00"},'
                ' {"oops": 1}, "junk"]')
    legacy = CommandStore(legacy_dir)
    assert len(legacy.list_commands()) == 1 and "id" in legacy.list_commands()[0]
    assert legacy.check_trigger("ping") == "pong"
    print("✓ history cap and legacy/malformed command files")


def run_all():
    print("=== MEMORY STORE TESTS ===")
    for test in (
        test_unique_ids_and_validation, test_dedupe_and_reinforcement, test_lifecycle,
        test_returns_copies_and_stats, test_reads_never_write, test_rollback_on_save_failure,
        test_corruption_recovery, test_atomic_save_edge_cases, test_thread_safety,
        test_journal_and_export, test_command_store, test_command_history_cap_and_legacy,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp)
    print("\n=== ALL MEMORY STORE TESTS PASSED ===")


if __name__ == "__main__":
    run_all()
