"""Architecture and persistence tests.

Covers the integration guarantees end to end on real on-disk stores:

1. commands persist without an explicit save and fire deterministically;
2. the runtime clock is injected into the prompt;
3. command history is surfaced to the model;
4. Elysium governing directives are answered by the application-level Elysium
   layer and are *not* injected into Astra's prompt, even when the state file
   appears after the orchestrator was constructed.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime

from astra.elysium import ElysiumOrchestrator
from astra.memory import CommandStore, TripleMemoryStore, atomic_save
from astra.orchestrator import CompanionOrchestrator


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.storage = os.path.join(self.root, "storage")
        self.config = os.path.join(self.root, "config")
        os.makedirs(self.config, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_command_persists_without_explicit_save(self):
        cmd_store = CommandStore(data_dir=self.storage)
        cmd_store.add_command("Next time I say echo, say bravo.", "echo", "bravo", once=True)
        del cmd_store  # no /exit required: the write already landed

        reloaded = CommandStore(data_dir=self.storage)
        self.assertEqual(reloaded.check_trigger("echo"), "bravo")

    def test_runtime_clock_in_prompt(self):
        store = TripleMemoryStore(data_dir=self.storage)
        astra = CompanionOrchestrator(store, config_dir=self.config)
        prompt = astra.build_prompt("What time is it?", [])
        self.assertIn("CURRENT SYSTEM DATE AND TIME:", prompt)
        self.assertIn(str(datetime.now().year), prompt)

    def test_command_history_in_prompt(self):
        cmd_store = CommandStore(data_dir=self.storage)
        cmd_store.add_command("Next time I say echo, say bravo.", "echo", "bravo")
        store = TripleMemoryStore(data_dir=self.storage)
        astra = CompanionOrchestrator(store, cmd_store, config_dir=self.config)
        prompt = astra.build_prompt("What time is it?", [])
        self.assertIn("DIRECT COMMAND HISTORY", prompt)
        self.assertIn("echo", prompt)
        self.assertIn("bravo", prompt)

    def test_elysium_directive_stays_out_of_prompt_when_added_after_construction(self):
        store = TripleMemoryStore(data_dir=self.storage)
        astra = CompanionOrchestrator(store, config_dir=self.config)

        # Directive appears only after the orchestrator was built.
        elysium = ElysiumOrchestrator(config_dir=self.config)
        elysium.state["active_directives"].append("Astra must speak formally.")
        atomic_save(elysium.state_file, elysium.state)

        prompt = astra.build_prompt("Hello", [])
        self.assertNotIn("Astra must speak formally.", prompt)
        self.assertNotIn("ELYSIUM ROOT GOVERNING", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
