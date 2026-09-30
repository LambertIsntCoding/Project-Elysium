"""Tests for the CLI driver (main.py).

Routing is exercised against a real loopback HTTP server and a real
TripleMemoryStore/CommandStore on disk, so no component is mocked.
"""

import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from elysium import CommandExtractor, ElysiumCommandRecorder, ElysiumOrchestrator
from main import ChatSession, build_session, looks_like_command_request
from memory_store import CommandStore, TripleMemoryStore
from orchestrator import CompanionOrchestrator


class _FakeOllama(BaseHTTPRequestHandler):
    response = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"response": json.dumps(type(self).response)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _StubStore:
    def get_active_memories(self, model):
        return []


class _DriverFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/api/generate"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _FakeOllama.response = {}
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.cmd_store = CommandStore(data_dir=self.tmp)
        self.elysium = ElysiumOrchestrator(config_dir=self.tmp, ollama_url=self.url)
        self.recorder = ElysiumCommandRecorder(self.cmd_store, CommandExtractor(self.url))
        self.orch = CompanionOrchestrator(
            self.store, config_dir=self.tmp, elysium=self.elysium, cmd_store=self.cmd_store
        )
        self.emitted = []
        self.session = ChatSession(
            self.orch, self.cmd_store, elysium=self.elysium, recorder=self.recorder,
            output_fn=self.emitted.append, input_fn=lambda _: "",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestHelpers(unittest.TestCase):
    def test_command_hint_detection(self):
        self.assertTrue(looks_like_command_request("Next time I say ping, say pong"))
        self.assertTrue(looks_like_command_request("when i say hi"))
        self.assertFalse(looks_like_command_request("what command did I give you?"))
        self.assertFalse(looks_like_command_request("just chatting"))


class TestRouting(_DriverFixture):
    def test_deterministic_command_bypasses_model(self):
        self.cmd_store.add_command("when I say ping", "ping", "pong")
        # No fake Ollama response set: a model call would yield "{}"/error.
        self.assertEqual(self.session.handle("ping"), "pong")
        self.assertFalse(any("Error" in line for line in self.emitted))

    def test_elysium_interception(self):
        _FakeOllama.response = {
            "system_response": "Directive added.",
            "new_directive_to_add": "always be concise",
            "directive_to_remove": None,
        }
        result = self.session.handle("elysium add a rule")
        self.assertTrue(result.startswith("ELYSIUM: Directive added."))
        self.assertTrue(any("Intercepted by ELYSIUM Root Layer" in line for line in self.emitted))
        self.assertEqual([d.content for d in self.elysium.get_active_directives()], ["always be concise"])

    def test_command_registration_validated(self):
        _FakeOllama.response = {"is_command": True, "confidence": 0.9,
                                "trigger": "ping", "response": "pong"}
        result = self.session.handle("next time I say ping, reply pong")
        self.assertEqual(result, "[System: Command registered.]")
        self.assertEqual(len(self.cmd_store.list_commands()), 1)

    def test_command_registration_rejects_unverifiable_trigger(self):
        _FakeOllama.response = {"is_command": True, "confidence": 0.9,
                                "trigger": "not-in-text", "response": "nope"}
        result = self.session.handle("next time I say ping, reply pong")
        # Not registered; falls through to a normal turn.
        self.assertNotEqual(result, "[System: Command registered.]")
        self.assertEqual(self.cmd_store.list_commands(), [])

    def test_normal_turn_calls_model(self):
        _FakeOllama.response = {}
        self.orch.query_gemma = lambda prompt, *a, **k: "hello there"
        self.assertEqual(self.session.handle("how are you?"), "hello there")
        self.assertEqual(self.session.turn_counter, 1)
        self.assertEqual(len(self.session.history), 2)


class TestSlashCommands(_DriverFixture):
    def test_memories_and_memory(self):
        mem_id = self.store.add_memory("roum", "Roum likes tea.", "explicit_fact",
                                       "explicit_user_statement")
        self.session._handle_slash("/memories")
        self.assertTrue(any("ROUM MODEL MEMORIES" in line for line in self.emitted))
        self.assertTrue(any(mem_id in line for line in self.emitted))

        self.emitted.clear()
        self.session._handle_slash(f"/memory {mem_id}")
        self.assertTrue(any("FULL RECORD" in line for line in self.emitted))

    def test_journal(self):
        self.store.add_journal_entry("Milestone", "Something happened.", "conv_x")
        self.session._handle_slash("/journal")
        self.assertTrue(any("AI MEMORY JOURNAL" in line for line in self.emitted))

    def test_unknown_slash(self):
        self.session._handle_slash("/nope")
        self.assertTrue(any("Unknown command" in line for line in self.emitted))

    def test_correction_supersedes_memory(self):
        mem_id = self.store.add_memory("roum", "Roum dislikes tea.", "explicit_fact",
                                       "explicit_user_statement")
        answers = iter([mem_id, "Roum likes tea.", "user corrected it"])
        self.session.input_fn = lambda _: next(answers)
        self.session._handle_slash("/correct")
        self.assertTrue(any("corrected successfully" in line for line in self.emitted))
        active = self.store.get_active_memories("roum")
        self.assertEqual([m["content"] for m in active], ["Roum likes tea."])


class TestSessionLoop(_DriverFixture):
    def test_exit_stops_loop(self):
        self.session.input_fn = lambda _: "/exit"
        self.session.run()
        self.assertTrue(any("Session ended cleanly" in line for line in self.emitted))

    def test_eof_stops_loop(self):
        def raise_eof(_):
            raise EOFError

        self.session.input_fn = raise_eof
        self.session.run()
        self.assertTrue(any("Exiting session" in line for line in self.emitted))


class TestBuildSession(unittest.TestCase):
    def test_build_session_defaults(self):
        tmp = tempfile.mkdtemp()
        try:
            session = build_session(config_dir=tmp, storage_dir=tmp)
            self.assertIsInstance(session, ChatSession)
            self.assertIsNotNone(session.recorder)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
