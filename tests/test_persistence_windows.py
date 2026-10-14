"""Regression tests for the Windows memory-persistence failure.

The observed runtime error was::

    PermissionError: [WinError 5] Access is denied:
    '...\\self_model.json.<rand>.tmp' -> '...\\self_model.json'

raised from ``memory.py``'s ``atomic_save`` while a ``turn_batch`` was exiting.
The demonstrated root cause is *concurrent writes to the same destination*: a
live turn's batched flush and the background consolidation thread both called
``atomic_save`` on ``self_model.json`` without a shared lock, so two
``os.replace`` calls overlapped on one path. POSIX ``rename`` tolerates that;
Windows ``MoveFileEx`` raises ``WinError 5`` for the loser.

These tests lock in the fix: writes are serialized in-process, the replace is
retried on a transient sharing violation, cross-process writers coordinate where
the platform allows, temp files are cleaned up without touching the destination,
a persistent failure is surfaced honestly, and the Discord path distinguishes a
persistence failure from a generation failure.

``os.replace`` is patched directly (the real code path is exercised; only the
OS call is simulated). Where a behaviour is Windows-specific (a genuine
``MoveFileEx`` sharing violation) the test states that limitation explicitly -
the same code path is driven by a simulated ``PermissionError`` here.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import astra.memory as mem
from astra.memory import (
    PersistenceError,
    TripleMemoryStore,
    TurnPersistenceError,
    atomic_save,
)
from astra import discord_adapter as da
from astra.memory import CommandStore
from astra.orchestrator import CompanionOrchestrator
from main import ChatSession


def _fast_retry():
    """Shrink the retry budget so persistent-failure tests do not sleep long."""
    return mock.patch.multiple(
        mem,
        REPLACE_MAX_ATTEMPTS=3,
        REPLACE_BACKOFF_BASE_SECONDS=0.001,
        REPLACE_BACKOFF_MAX_SECONDS=0.005,
    )


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


class TestAtomicSaveReplaceRetry(unittest.TestCase):
    """The bounded retry around the atomic replace."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "self_model.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_transient_replace_failure_recovers(self):
        atomic_save(self.path, [{"id": "old"}])
        calls = {"n": 0}
        real = os.replace

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(13, "Access is denied")
            return real(src, dst)

        with _fast_retry(), mock.patch("os.replace", side_effect=flaky):
            atomic_save(self.path, [{"id": "new"}])

        self.assertGreaterEqual(calls["n"], 3)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"id": "new"}])
        # The transient failure left no temp file behind.
        self.assertEqual([n for n in os.listdir(self.tmp) if n.endswith(".tmp")], [])

    def test_persistent_replace_failure_is_surfaced_honestly(self):
        atomic_save(self.path, [{"id": "keep"}])
        before = _read_bytes(self.path)

        def always_denied(src, dst):
            raise PermissionError(13, "Access is denied")

        with _fast_retry(), mock.patch("os.replace", side_effect=always_denied):
            with self.assertRaises(PermissionError) as ctx:
                atomic_save(self.path, [{"id": "doomed"}])
        # The original error is preserved, not replaced by a generic message.
        self.assertIn("Access is denied", str(ctx.exception))

        # The previous valid JSON is untouched and no temp file is left behind.
        self.assertEqual(_read_bytes(self.path), before)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.endswith(".tmp")], [])

    def test_failed_write_does_not_delete_the_destination(self):
        atomic_save(self.path, [{"id": "precious"}])
        with _fast_retry(), mock.patch(
                "os.replace", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(PermissionError):
                atomic_save(self.path, [{"id": "other"}])
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"id": "precious"}])


class TestStoreWriteSerialization(unittest.TestCase):
    """In-process writers must never overlap ``os.replace`` on one file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.mid = self.store.add_memory(
            "roum", "seed fact", "explicit_fact", "explicit_user_statement")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _overlap_detector(self):
        state = {"inside": 0, "overlap": 0}
        guard = threading.Lock()
        real = mem.atomic_save

        def spy(path, *a, **k):
            if os.path.basename(path) == "self_model.json":
                with guard:
                    state["inside"] += 1
                    if state["inside"] > 1:
                        state["overlap"] += 1
                try:
                    return real(path, *a, **k)
                finally:
                    with guard:
                        state["inside"] -= 1
            return real(path, *a, **k)

        return state, spy

    def test_turn_batch_and_background_write_do_not_overlap(self):
        # Reproduces the reported failure: a live turn's batch exit and the
        # background consolidation thread both write self_model.json.
        state, spy = self._overlap_detector()

        def turn():
            for _ in range(20):
                with self.store.turn_batch():
                    self.store.apply_live_affect_event(
                        {"kind": "happy", "intensity": 0.5})

        def consolidate():
            for i in range(20):
                self.store.add_memory("self", f"observation {i}",
                                      "self_observation", "ai_extraction")

        with mock.patch.object(mem, "atomic_save", side_effect=spy):
            ta = threading.Thread(target=turn)
            tb = threading.Thread(target=consolidate)
            ta.start()
            tb.start()
            ta.join()
            tb.join()

        self.assertEqual(state["overlap"], 0,
                         "writers overlapped os.replace on the same file")

    def test_concurrent_writers_lose_no_memories(self):
        def worker(n):
            for i in range(20):
                self.store.add_memory("roum", f"thread {n} fact {i}",
                                      "explicit_fact", "t")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        reopened = TripleMemoryStore(data_dir=self.tmp)
        contents = {m["content"] for m in reopened.get_active_memories("roum")}
        for n in range(6):
            for i in range(20):
                self.assertIn(f"thread {n} fact {i}", contents)


class TestStorePersistenceFailureReporting(unittest.TestCase):
    """A failed save is reported and the in-memory state is preserved."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.mid = self.store.add_memory("roum", "keep me", "explicit_fact", "t")
        self.path = os.path.join(self.tmp, "roum_model.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_batch_exit_failure_raises_and_preserves_disk(self):
        before = _read_bytes(self.path)
        with _fast_retry(), mock.patch.object(
                mem, "atomic_save", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(PersistenceError) as ctx:
                with self.store.turn_batch():
                    self.store.record_use([self.mid])

        self.assertIn("roum", ctx.exception.failed)
        self.assertEqual(_read_bytes(self.path), before)
        # The failed key is reported and stays dirty so a later flush retries.
        self.assertIn("roum", self.store.last_persist_error or {})
        self.assertIn("roum", self.store._dirty)

    def test_a_later_flush_recovers_the_pending_write(self):
        with _fast_retry(), mock.patch.object(
                mem, "atomic_save", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(PersistenceError):
                with self.store.turn_batch():
                    self.store.record_use([self.mid])

        # Once the transient condition clears, the retry persists the change.
        self.store.flush()
        self.assertIsNone(self.store.last_persist_error)
        self.assertNotIn("roum", self.store._dirty)
        reopened = TripleMemoryStore(data_dir=self.tmp)
        self.assertGreaterEqual(reopened.get_memory("roum", self.mid)["use_count"], 1)

    def test_failure_is_not_reported_as_success(self):
        denied = PermissionError(13, "Access is denied")
        with _fast_retry(), mock.patch.object(mem, "atomic_save", side_effect=denied):
            with self.assertRaises(PersistenceError):
                with self.store.turn_batch():
                    self.store.record_use([self.mid])
        self.assertTrue(self.store.last_persist_error)
        # While the condition persists, a retry must fail again - never a
        # silent "saved" after a failed write.
        with _fast_retry(), mock.patch.object(mem, "atomic_save", side_effect=denied):
            with self.assertRaises(PersistenceError):
                self.store.flush()
        self.assertTrue(self.store.last_persist_error)


class TestCrossProcessWriters(unittest.TestCase):
    """Two processes may legitimately share a store directory.

    On Linux ``os.replace`` is atomic, so this asserts no torn/interleaved file
    and no leftover temp files; the advisory lock is what makes the equivalent
    safe on Windows, where a genuine ``MoveFileEx`` sharing violation cannot be
    reproduced off-Windows. That limitation is stated rather than hidden.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "self_model.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_concurrent_processes_leave_one_valid_file(self):
        payload_a = [{"writer": "a", "n": i} for i in range(400)]
        payload_b = [{"writer": "b", "n": i} for i in range(400)]
        script = (
            "import sys, astra.memory as m\n"
            "import json\n"
            "payload = json.loads(sys.argv[2])\n"
            "for _ in range(30):\n"
            "    m.atomic_save(sys.argv[1], payload)\n"
        )
        procs = [
            subprocess.Popen([sys.executable, "-c", script, self.path,
                              json.dumps(payload)],
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for payload in (payload_a, payload_b)
        ]
        for proc in procs:
            self.assertEqual(proc.wait(timeout=60), 0)

        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)  # must be valid JSON, never torn
        self.assertIn(data, (payload_a, payload_b))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.endswith(".tmp")], [])

    def test_advisory_lock_excludes_a_second_process(self):
        # While this process holds the sidecar lock, another process must not
        # take it - that is the cross-process coordination that keeps two
        # writers off the same destination on Windows.
        holder = mem._acquire_file_lock(self.path)
        self.assertIsNotNone(holder)
        script = (
            "import sys, astra.memory as m\n"
            "m.LOCK_ACQUIRE_TIMEOUT = 0.2\n"
            "fd = m._acquire_file_lock(sys.argv[1])\n"
            "print('acquired' if fd is not None else 'blocked')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script, self.path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=60)
        mem._release_file_lock(holder)
        self.assertEqual(out.stdout.strip(), "blocked")


# ---------------------------------------------------------------------
# The Discord / session error path
# ---------------------------------------------------------------------
class _SessionFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.cmd_store = CommandStore(data_dir=self.tmp)
        self.orch = CompanionOrchestrator(
            self.store, config_dir="./config", cmd_store=self.cmd_store)
        self.session = ChatSession(
            self.orch, self.cmd_store, output_fn=lambda _t: None,
            input_fn=lambda _p: "")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSessionPersistenceFailure(_SessionFixture):
    def test_generated_reply_with_failed_save_raises_persistence_error(self):
        self.session.orchestrator.query_gemma = lambda prompt: "a real reply"
        with _fast_retry(), mock.patch.object(
                mem, "atomic_save", side_effect=PermissionError(13, "Access is denied")):
            with self.assertRaises(TurnPersistenceError) as ctx:
                self.session.handle("hello")

        # The reply was generated: it is carried, and this is NOT a generation
        # failure. The failed model keys are reported.
        self.assertEqual(ctx.exception.response, "a real reply")
        self.assertTrue(ctx.exception.failed)
        self.assertNotIsInstance(ctx.exception, type(None))

    def test_successful_turn_is_not_affected(self):
        self.session.orchestrator.query_gemma = lambda prompt: "a real reply"
        self.assertEqual(self.session.handle("hello"), "a real reply")


class _DiscordFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _adapter(self, transport, router):
        return da.DiscordAdapter(transport, router, channel_id="c1",
                                 data_dir=self.tmp, require_mention=True,
                                 bot_user_id="42")


class _Transport:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []
        self.typing_calls = []

    def fetch_messages(self, channel_id, *, after=None, limit=20):
        msgs = list(self.messages)
        if after:
            msgs = [m for m in msgs if str(m["id"]) > str(after)]
        return {"ok": True, "data": msgs[::-1][:limit]}

    def send_message(self, channel_id, content, *, reference=None,
                     mention_user_id=None):
        self.sent.append({"channel": channel_id, "content": content})
        return {"ok": True, "data": {"id": "srv-1"}}

    def send_typing(self, channel_id):
        self.typing_calls.append(channel_id)
        return {"ok": True, "data": {}}


class _PersistFailRouter:
    """A router whose turn generated a reply but failed to persist memory."""

    def __init__(self, reply="a real reply"):
        self.reply = reply
        self.calls = []

    def handle(self, text, *, source="cli"):
        self.calls.append((text, source))
        raise TurnPersistenceError("could not persist self",
                                   failed={"self": "Access is denied"},
                                   response=self.reply)


class _GenFailRouter:
    def handle(self, text, *, source="cli"):
        raise RuntimeError("model unavailable")


def _mention(mid, content="<@42> hi"):
    return {"id": mid, "content": content, "author": {"id": "u1"},
            "channel_id": "c1"}


class TestDiscordPersistenceFailure(_DiscordFixture):
    def test_reply_delivered_and_recorded_unsaved_not_generation_failed(self):
        transport = _Transport([_mention("1")])
        adapter = self._adapter(transport, _PersistFailRouter())
        adapter.poll_once()  # baseline
        transport.messages = [_mention("2", "<@42> hello")]
        adapter.poll_once()

        # The generated reply was delivered...
        self.assertEqual(transport.sent[-1]["content"], "a real reply")
        # ...and recorded honestly as delivered-but-not-durable, not answered.
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_UNSAVED)
        self.assertEqual(adapter.persist_failed, 1)
        self.assertEqual(adapter.delivered, 1)
        self.assertEqual(adapter.failed, 0)
        self.assertNotIn("generation failed", adapter.last_error or "")
        self.assertIn("memory not persisted", adapter.last_error or "")

    def test_generation_failure_still_recorded_as_failed(self):
        transport = _Transport([_mention("1")])
        adapter = self._adapter(transport, _GenFailRouter())
        adapter.poll_once()  # baseline
        transport.messages = [_mention("2", "<@42> hello")]
        adapter.poll_once()

        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_FAILED)
        self.assertIn("generation failed", adapter.last_error or "")
        self.assertEqual(adapter.persist_failed, 0)
        self.assertEqual(transport.sent, [])

    def test_unsaved_message_is_retried_without_duplicating_the_reply(self):
        transport = _Transport([_mention("1")])
        adapter = self._adapter(transport, _PersistFailRouter())
        adapter.poll_once()  # baseline
        transport.messages = [_mention("2", "<@42> hello")]
        adapter.poll_once()
        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_UNSAVED)
        self.assertEqual(len(transport.sent), 1)
        self.assertTrue(adapter.inbox.was_replied("2"))

        # An unsaved turn stays unresolved, so once it is old enough it is
        # retried; this time persistence succeeds and it is marked answered -
        # but the reply is NOT sent a second time (duplicate side effect guard).
        import astra.discord_adapter as dmod
        adapter._now = lambda: (
            dmod.datetime.now(dmod.timezone.utc)
            + dmod.timedelta(seconds=dmod.RECOVER_MIN_AGE_SECONDS + 1))
        adapter.router = _OkRouter("a real reply")
        adapter.poll_once()

        self.assertEqual(adapter.inbox.state_of("2"), da.MSG_RESPONDED)
        self.assertEqual(len(transport.sent), 1)

    def test_diagnostics_report_persistence_failures(self):
        transport = _Transport([_mention("1")])
        adapter = self._adapter(transport, _PersistFailRouter())
        adapter.poll_once()
        transport.messages = [_mention("2", "<@42> hello")]
        adapter.poll_once()
        diag = adapter.diagnostics()
        self.assertEqual(diag["inbound_persist_failed"], 1)
        self.assertIn("memory not persisted", diag["last_error"] or "")


class _OkRouter:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def handle(self, text, *, source="cli"):
        self.calls.append((text, source))
        return {"text": self.reply, "handled": True, "source": source}


class TestMemoryConsistencyOnRetry(unittest.TestCase):
    """A retried turn must not duplicate an identical memory record."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reprocessing_the_same_statement_dedupes(self):
        first = self.store.add_memory("roum", "Roum likes strong coffee",
                                      "explicit_preference", "explicit_user_statement")
        second = self.store.add_memory("roum", "Roum likes strong coffee",
                                       "explicit_preference", "explicit_user_statement")
        self.assertEqual(first, second)
        self.assertEqual(len(self.store.get_active_memories("roum")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
