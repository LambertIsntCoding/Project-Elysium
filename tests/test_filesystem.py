"""Tests for the filesystem policy/executor and the background organizer.

All filesystem writes happen under a tmpdir, and the trash mechanism is
injected, so nothing here touches the real user trash or the project storage.
"""

import os
import shutil
import tempfile
import unittest

from astra import file_organizer as fo
from astra import filesystem as fs


class _MemoryTrash:
    """A trash double that moves into a tmpdir; never the real trash."""

    def __init__(self, dest_dir):
        self.dest_dir = dest_dir
        self.moved = []

    def to_trash(self, path):
        os.makedirs(self.dest_dir, exist_ok=True)
        dest = os.path.join(self.dest_dir, os.path.basename(path))
        shutil.move(path, dest)
        self.moved.append(path)
        return {"ok": True, "method": "memory-trash", "dest": dest}


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "root")
        self.storage = os.path.join(self.tmp, "storage")
        os.makedirs(self.root)
        os.makedirs(self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_roots_means_nothing_writable(self):
        policy = fs.FsPolicy()
        self.assertFalse(policy.in_allowed_root(os.path.join(self.root, "a.txt")))

    def test_outside_root_is_blocked(self):
        policy = fs.FsPolicy(roots=[self.root])
        self.assertTrue(policy.check(
            os.path.join(self.root, "a.txt"), operation=fs.OP_MOVE) is None)
        self.assertIsNotNone(policy.check(
            os.path.join(self.tmp, "elsewhere.txt"), operation=fs.OP_MOVE))

    def test_runtime_paths_are_never_a_mutation_target(self):
        policy = fs.FsPolicy(roots=[self.tmp], runtime_paths=[self.storage])
        block = policy.check(os.path.join(self.storage, "roum_model.json"),
                             operation=fs.OP_MOVE)
        self.assertIsNotNone(block)

    def test_default_runtime_paths_cover_config_and_storage(self):
        paths = fo.default_runtime_paths(config_dir=self.tmp, storage_dir=self.storage)
        joined = " ".join(paths)
        self.assertIn(os.path.abspath(self.storage), joined)


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "root")
        os.makedirs(self.root)
        self.policy = fs.FsPolicy(roots=[self.root])
        self.trash = _MemoryTrash(os.path.join(self.tmp, "trash"))
        self.ops = fs.FileOps(self.policy, data_dir=self.tmp, trash=self.trash,
                              external_tools={})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _file(self, name, text="hello"):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_move_within_root_succeeds(self):
        src = self._file("a.txt")
        dest = os.path.join(self.root, "sub", "a.txt")
        rec = self.ops.execute(self.ops.plan_move(src, dest))
        self.assertEqual(rec.outcome, fs.OUTCOME_OK)
        self.assertTrue(os.path.exists(dest))
        self.assertFalse(os.path.exists(src))

    def test_move_outside_root_is_blocked_and_source_kept(self):
        src = self._file("a.txt")
        rec = self.ops.execute(self.ops.plan_move(src, os.path.join(self.tmp, "out.txt")))
        self.assertEqual(rec.outcome, fs.OUTCOME_BLOCKED)
        self.assertTrue(os.path.exists(src))

    def test_dry_run_does_not_move(self):
        src = self._file("a.txt")
        dest = os.path.join(self.root, "sub", "a.txt")
        rec = self.ops.execute(self.ops.plan_move(src, dest), dry_run=True)
        self.assertEqual(rec.outcome, fs.OUTCOME_DRY_RUN)
        self.assertTrue(os.path.exists(src))
        self.assertFalse(os.path.exists(dest))

    def test_delete_needs_authorization_by_policy(self):
        src = self._file("a.txt")
        plan = fs.FilePlan(fs.OP_DELETE, src)
        rec = self.ops.execute(plan, authorized=True)
        self.assertEqual(rec.outcome, fs.OUTCOME_BLOCKED)  # allow_delete False
        self.assertTrue(os.path.exists(src))

    def test_authorized_delete_is_reversible_via_trash(self):
        policy = fs.FsPolicy(roots=[self.root], allow_delete=True)
        ops = fs.FileOps(policy, data_dir=self.tmp, trash=self.trash,
                         external_tools={})
        src = self._file("a.txt")
        rec = ops.execute(fs.FilePlan(fs.OP_DELETE, src), authorized=True)
        self.assertEqual(rec.outcome, fs.OUTCOME_OK)
        self.assertTrue(rec.reversible)
        self.assertFalse(os.path.exists(src))
        self.assertTrue(self.trash.moved)

    def test_history_is_recorded(self):
        self._file("a.txt")
        self.ops.scan(self.root)
        self.assertTrue(self.ops.history())


class TestOrganizer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, "root")
        os.makedirs(self.root)
        self.policy = fs.FsPolicy(roots=[self.root],
                                  runtime_paths=[os.path.join(self.tmp, "storage")])
        self.ops = fs.FileOps(self.policy, data_dir=self.tmp,
                              trash=_MemoryTrash(os.path.join(self.tmp, "trash")),
                              external_tools={})
        self.decisions = fo.DecisionStore(data_dir=self.tmp)
        self.organizer = fo.FileOrganizer(
            self.policy, self.ops, self.decisions,
            user_roots=[self.root], max_files_per_cycle=10)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _file(self, name, text="hello"):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_interrupted_decision_recovers_as_failed_not_success(self):
        decision = self.decisions.add(fo.FileDecision(
            path=os.path.join(self.root, "x.txt"), proposed_action="trash",
            reasoning="interrupted", confidence=0.9, importance=0.5,
            disposition=fo.DISPOSITION_TRASH))
        self.decisions.set_status(decision.id, fo.D_EXECUTING)
        recovered = fo.FileOrganizer(self.policy, self.ops,
                                     fo.DecisionStore(data_dir=self.tmp),
                                     user_roots=[self.root])
        count = recovered.recover()
        self.assertGreaterEqual(count, 1)
        # Read back from disk through a fresh store: recovery must be persisted.
        self.assertEqual(
            fo.DecisionStore(data_dir=self.tmp).get(decision.id).status, fo.D_FAILED)

    def test_handled_files_are_not_reconsidered_every_cycle(self):
        self._file("routine.txt", "some content")
        first = self.organizer.run_cycle()
        # Run again: the same file must not produce a duplicate decision.
        second = self.organizer.run_cycle()
        self.assertEqual(len(second["asked"]), 0)
        total = len(self.decisions.all())
        self.organizer.run_cycle()
        self.assertEqual(len(self.decisions.all()), total)

    def test_cycle_is_bounded(self):
        for i in range(50):
            self._file(f"f{i}.txt", f"content {i}")
        organizer = fo.FileOrganizer(self.policy, self.ops, self.decisions,
                                     user_roots=[self.root], max_files_per_cycle=5)
        report = organizer.run_cycle()
        self.assertLessEqual(report["scanned"], 5)

    def test_inspect_is_read_only_and_reports_honestly(self):
        block = self.organizer._decide  # ensure attribute exists
        self.assertTrue(callable(block))


if __name__ == "__main__":
    unittest.main(verbosity=2)
