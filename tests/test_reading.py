"""Tests for background reading (Slice 3).

Everything runs against real code paths on real temp directories:

* a real :class:`~astra.library.Library` (text and EPUB ingestion, atomic
  resume);
* a real :class:`~astra.memory.TripleMemoryStore` (the reading results become
  ordinary memories);
* a real :class:`~astra.reader.BackgroundReader` whose only stand-in is the
  model call, which is injected by design.

The behaviours under test are the ones the brief makes non-negotiable: reading
yields to interaction and to games, it is resumable, it does not spend a model
call per fragment, missing context stays an open question rather than becoming a
"fact", a revision is an inference that cannot override an explicit memory, and
the prompt stays read-only.
"""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
import zipfile

from astra import affect
from astra import inquiry
from astra import reading
from astra.library import Library, epub_to_text, html_to_text, slugify
from astra.memory import TripleMemoryStore
from astra.orchestrator import CompanionOrchestrator
from astra.reader import BackgroundReader


def _hashes(directory):
    out = {}
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, directory)] = hashlib.sha256(handle.read()).hexdigest()
    return out


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)
        self.library = Library(data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def reader(self, model=None, **kwargs):
        kwargs.setdefault("cpu_load_fn", lambda: 0.0)
        kwargs.setdefault("process_names_fn", lambda: [])
        kwargs.setdefault("idle_seconds", 0.0)
        return BackgroundReader(self.library, self.store, model or (lambda p: "{}"), **kwargs)


# ---------------------------------------------------------------------
# The gate: reading yields to interaction, games, and load
# ---------------------------------------------------------------------
class TestReaderGate(unittest.TestCase):
    def test_game_always_wins_over_idle(self):
        allowed, reason = reading.should_read(seconds_since_input=10_000, game="dota2.exe")
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_GAME_RUNNING)

    def test_gpu_busy_counts_as_game(self):
        allowed, reason = reading.should_read(seconds_since_input=10_000, gpu_busy=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_GAME_RUNNING)

    def test_recent_interaction_pauses(self):
        allowed, reason = reading.should_read(seconds_since_input=5, idle_threshold=90)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_USER_ACTIVE)

    def test_idle_and_calm_allows(self):
        allowed, reason = reading.should_read(seconds_since_input=500, cpu_load=0.2)
        self.assertTrue(allowed)
        self.assertEqual(reason, reading.IDLE_OK)

    def test_machine_load_stands_down(self):
        allowed, reason = reading.should_read(seconds_since_input=500, cpu_load=0.95,
                                              load_ceiling=0.75)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_MACHINE_BUSY)

    def test_game_beats_load(self):
        # A game is reported ahead of a generic "machine busy".
        _, reason = reading.should_read(seconds_since_input=10_000, game="steam.exe",
                                        cpu_load=0.99)
        self.assertEqual(reason, reading.IDLE_GAME_RUNNING)

    def test_recent_interaction_beats_load(self):
        # Roum actively interacting is reported ahead of machine load, so the
        # reason Astra is quiet reflects what actually interrupted her.
        _, reason = reading.should_read(seconds_since_input=1, cpu_load=0.99)
        self.assertEqual(reason, reading.IDLE_USER_ACTIVE)

    def test_no_work_is_not_an_error(self):
        allowed, reason = reading.should_read(seconds_since_input=500, has_work=False)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_NO_WORK)

    def test_disabled_short_circuits(self):
        allowed, reason = reading.should_read(enabled=False)
        self.assertFalse(allowed)
        self.assertEqual(reason, reading.IDLE_DISABLED)

    def test_game_detection_is_specific(self):
        self.assertEqual(reading.looks_like_game(["chrome.exe", "steam.exe"]), "steam.exe")
        # A browser is not a game just because the title might mention one.
        self.assertIsNone(reading.looks_like_game(["chrome.exe", "explorer.exe"]))
        self.assertIsNone(reading.looks_like_game([]))


# ---------------------------------------------------------------------
# Ingestion and resumable position
# ---------------------------------------------------------------------
class TestLibrary(_Base):
    def test_text_ingestion_and_cache(self):
        state = self.library.add_text("hello world. " * 50, title="Hello", work_id="hello")
        self.assertEqual(state["work_id"], "hello")
        self.assertGreater(state["total_units"], 0)
        self.assertEqual(self.library.current_work_id(), "hello")
        self.assertIn("hello world", self.library.text_for("hello"))

    def test_epub_ingestion_stdlib_only(self):
        path = os.path.join(self.tmp, "book.epub")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
                '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                '</rootfiles></container>')
            archive.writestr(
                "OEBPS/content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
                '<manifest><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/>'
                '<item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest>'
                '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>')
            archive.writestr("OEBPS/a.xhtml",
                             "<html><head><style>x{}</style></head><body><h1>One</h1>"
                             "<p>Alice found a key.</p><script>bad()</script></body></html>")
            archive.writestr("OEBPS/b.xhtml",
                             "<html><body><h1>Two</h1><p>She opened the gate.</p></body></html>")
        state = self.library.add_file(path)
        text = self.library.text_for(state["work_id"])
        self.assertEqual(state["source_format"], reading.SOURCE_FORMAT_EPUB)
        self.assertIn("Alice found a key.", text)
        self.assertIn("She opened the gate.", text)
        # Head/script/style content is stripped.
        self.assertNotIn("bad()", text)
        self.assertNotIn("x{}", text)
        # Reading order follows the spine.
        self.assertLess(text.index("Alice"), text.index("gate"))

    def test_html_to_text_keeps_paragraphs(self):
        text = html_to_text("<p>One</p><p>Two</p>")
        self.assertIn("One", text)
        self.assertIn("Two", text)
        self.assertIn("\n", text)

    def test_position_survives_reload(self):
        self.library.add_text("abcdefghij " * 100, title="Alpha", work_id="alpha")
        state = self.library.get_state("alpha")
        state = state | {"char_offset": 250, "units_read": 4}
        self.library.save_state("alpha", state)
        del self.library

        reopened = Library(data_dir=self.tmp)
        resumed = reopened.get_state("alpha")
        self.assertEqual(resumed["char_offset"], 250)
        self.assertEqual(resumed["units_read"], 4)
        self.assertEqual(reopened.current_work_id(), "alpha")

    def test_reingest_does_not_clobber_position(self):
        self.library.add_text("body " * 100, title="Same", work_id="same")
        self.library.save_state("same", self.library.get_state("same") | {"char_offset": 42})
        # Adding another work must not reset the first one's position.
        self.library.add_text("other " * 100, title="Other", work_id="other")
        self.assertEqual(self.library.get_state("same")["char_offset"], 42)

    def test_slugify_is_stable_and_safe(self):
        self.assertEqual(slugify("The Sea, and the Garden!"), "the-sea-and-the-garden")
        self.assertEqual(slugify(""), "work")


# ---------------------------------------------------------------------
# Chunking: bounded, coherent, and no model call per fragment
# ---------------------------------------------------------------------
class TestChunking(unittest.TestCase):
    def test_chunks_are_bounded_and_cover_everything(self):
        text = "Sentence number one. " * 2000
        total, pos, chunks = len(text), 0, 0
        while pos < total:
            start, end = reading.chunk_bounds(text, pos, max_chars=1000)
            self.assertLessEqual(end - start, 1000 + 1)
            pos, chunks = end, chunks + 1
        self.assertEqual(pos, total)
        self.assertGreater(chunks, 1)

    def test_chunk_ends_on_a_sentence_boundary(self):
        text = ("Alpha bravo charlie delta. " * 100)
        start, end = reading.chunk_bounds(text, 0, max_chars=300)
        self.assertTrue(text[start:end].rstrip().endswith("."))

    def test_short_tail_is_not_a_separate_call(self):
        text = "x" * 500 + " end."
        start, end = reading.chunk_bounds(text, 0, max_chars=490)
        # The tail is folded in rather than left as a 4-character chunk.
        self.assertEqual(end, len(text))

    def test_chunk_count_matches_manual_walk(self):
        text = "abc. " * 500
        self.assertEqual(reading.chunk_count(text, max_chars=200),
                         len(list(self._walk(text, 200))))

    @staticmethod
    def _walk(text, size):
        pos = 0
        while pos < len(text):
            start, end = reading.chunk_bounds(text, pos, max_chars=size)
            if end <= start:
                break
            yield start, end
            pos = end


# ---------------------------------------------------------------------
# Extraction parsing: tolerant, bounded, and never invents structure
# ---------------------------------------------------------------------
class TestExtractionParsing(unittest.TestCase):
    def test_parses_json_wrapped_in_prose(self):
        raw = 'Sure! Here you go:\n```json\n{"events": ["a"], "questions": "why?"}\n```'
        digest = reading.parse_extraction(raw)
        self.assertEqual(digest["events"], ["a"])
        self.assertEqual(digest["questions"], ["why?"])

    def test_bad_reply_costs_nothing(self):
        digest = reading.parse_extraction("not json at all")
        self.assertEqual(digest["events"], [])
        self.assertEqual(set(digest), {
            "entities", "events", "relationships", "ideas", "observations",
            "interpretations", "questions", "associations"})

    def test_entity_dicts_become_readable_lines(self):
        digest = reading.parse_extraction(
            '{"entities": [{"name": "Alice", "note": "protagonist"}]}')
        self.assertEqual(reading.item_text(digest["entities"][0]), "Alice: protagonist")

    def test_resume_context_is_bounded(self):
        digest = {"events": ["a" * 900], "ideas": ["b" * 900]}
        self.assertLessEqual(len(reading.resume_context_from(digest)), 600)


# ---------------------------------------------------------------------
# The reader cycle: bounded work, resume, pause, and no skip on failure
# ---------------------------------------------------------------------
class TestReaderCycle(_Base):
    def _digest_model(self):
        calls = []

        def model(prompt):
            calls.append(prompt)
            return json.dumps({
                "entities": [{"name": "Alice", "note": "protagonist"}],
                "events": ["Alice found a key"],
                "ideas": ["The gate is a threshold"],
                "observations": ["A brass key lay under the gate"],
                "interpretations": ["The key likely matters later"],
                "questions": ["Who left the key there?"],
            })

        return model, calls

    def test_one_cycle_is_bounded_and_saved(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        model, calls = self._digest_model()
        reader = self.reader(model, chunks_per_cycle=1)
        report = reader.run_once()
        self.assertTrue(report["read"])
        self.assertEqual(report["chunks"], 1)
        self.assertEqual(len(calls), 1)  # one model call, not one per sentence
        state = self.library.get_state("garden")
        self.assertGreater(state["char_offset"], 0)
        self.assertEqual(state["chunks"], 1)

    def test_resume_continues_from_saved_offset(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        model, _ = self._digest_model()
        self.reader(model).run_once()
        first = self.library.get_state("garden")["char_offset"]
        # A brand-new reader on the same data dir must not restart.
        fresh = self.reader(model)
        fresh.run_once()
        second = self.library.get_state("garden")["char_offset"]
        self.assertGreater(second, first)

    def test_game_stops_a_cycle_between_chunks(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        model, calls = self._digest_model()
        reader = self.reader(model, chunks_per_cycle=5)
        reader.note_game_pause("dota2.exe")
        report = reader.run_once()
        self.assertFalse(report["read"])
        self.assertEqual(report["reason"], reading.IDLE_GAME_RUNNING)
        self.assertEqual(len(calls), 0)

    def test_interaction_stops_a_cycle(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        model, calls = self._digest_model()
        reader = self.reader(model, idle_seconds=1000)
        reader.note_activity()
        report = reader.run_once()
        self.assertFalse(report["read"])
        self.assertEqual(report["reason"], reading.IDLE_USER_ACTIVE)
        self.assertEqual(len(calls), 0)

    def test_failed_model_call_does_not_skip_the_passage(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        attempts = {"n": 0}

        def flaky(prompt):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("backend down")
            return "{}"

        reader = self.reader(flaky)
        report = reader.run_once()
        self.assertFalse(report["read"])
        self.assertEqual(report["reason"], reading.IDLE_MODEL_FAILED)
        self.assertEqual(self.library.get_state("garden")["char_offset"], 0)
        # The next cycle retries the same passage rather than skipping it.
        self.reader(flaky).run_once()
        self.assertGreater(self.library.get_state("garden")["char_offset"], 0)

    def test_continuity_context_is_carried_into_the_next_prompt(self):
        self.library.add_text("Alice found a key. " * 500, title="Garden", work_id="garden")
        prompts = []

        def model(prompt):
            prompts.append(prompt)
            return json.dumps({"events": ["Alice found a key"],
                               "ideas": ["The gate is a threshold"]})

        self.reader(model, chunks_per_cycle=2).run_once()
        self.assertEqual(len(prompts), 2)
        # The first chunk starts cold; the second is read in sequence, carrying
        # what happened just before rather than being treated as an excerpt.
        self.assertNotIn("continuity", prompts[0])
        self.assertIn("continuity", prompts[1])

    def test_reading_to_the_end_marks_the_work_finished(self):
        self.library.add_text("Short passage. " * 30, title="Tiny", work_id="tiny")
        model, _ = self._digest_model()
        reader = self.reader(model, chunks_per_cycle=50)
        reader.run_once()
        self.assertEqual(self.library.get_state("tiny")["status"], reading.WORK_FINISHED)

    def test_nothing_queued_is_reported_not_crashed(self):
        reader = self.reader()
        report = reader.run_once()
        self.assertFalse(report["read"])
        self.assertEqual(report["reason"], reading.IDLE_NO_WORK)

    def test_daemon_start_stop_is_clean(self):
        self.library.add_text("Alice found a key. " * 50, title="Garden", work_id="garden")
        reader = self.reader(poll_seconds=0.01, chunks_per_cycle=1)
        reader.start()
        self.assertTrue(reader.running)
        reader.stop()
        self.assertFalse(reader.running)


# ---------------------------------------------------------------------
# Reading produces governed memories, not a parallel belief store
# ---------------------------------------------------------------------
class TestReadingMemories(_Base):
    DIGEST = {
        "entities": [{"name": "Alice", "note": "protagonist"}],
        "events": ["Alice found a key"],
        "observations": ["A brass key lay under the gate"],
        "ideas": ["The gate is a threshold"],
        "interpretations": ["The key likely matters later"],
        "questions": ["Who left the key there?"],
    }

    def test_results_are_ordinary_scoped_memories(self):
        result = self.store.apply_reading_results(
            work_id="garden", title="Garden", digest=self.DIGEST)
        self.assertGreater(result["memories"], 0)
        context = self.store.get_work_context("garden")
        self.assertTrue(context)
        self.assertTrue(all(inquiry.work_of(m) == "garden" for m in context))
        # Observations, interpretations and hypotheses all landed as such.
        types = {m["type"] for m in context}
        self.assertIn(inquiry.TYPE_OBSERVATION, types)
        self.assertIn(inquiry.TYPE_INTERPRETATION, types)

    def test_work_knowledge_never_leaks_across_works(self):
        self.store.apply_reading_results(work_id="garden", title="Garden", digest=self.DIGEST)
        self.store.apply_reading_results(
            work_id="sea", title="Sea",
            digest={"observations": ["The captain watched the grey water"]})
        garden = {m["content"] for m in self.store.get_work_context("garden")}
        sea = {m["content"] for m in self.store.get_work_context("sea")}
        self.assertTrue(garden.isdisjoint(sea))

    def test_unresolved_material_stays_an_open_question(self):
        self.store.apply_reading_results(work_id="garden", title="Garden", digest=self.DIGEST)
        questions = self.store.get_questions(open_only=True)
        self.assertTrue(any("key" in q["content"] for q in questions))
        for question in questions:
            self.assertEqual(inquiry.epistemic_of(question), inquiry.EPISTEMIC_UNKNOWN)

    def test_reading_inference_cannot_override_an_explicit_memory(self):
        # Roum states something explicitly; the reader later "interprets" the
        # opposite. The explicit memory must survive, and the reading note must
        # be filed as an inference (which weakens, never supersedes).
        explicit = self.store.add_memory(
            "roum", "Alice is the narrator.", "explicit_fact",
            "explicit_user_statement",
        )
        self.store.apply_reading_results(
            work_id="garden", title="Garden",
            digest={"interpretations": ["Alice is not the narrator"]})
        surviving = self.store.get_memory("roum", explicit)
        self.assertEqual(surviving["status"], "active")
        self.assertEqual(surviving["type"], "explicit_fact")

    def test_reading_records_an_experience_not_a_trait(self):
        self.store.apply_reading_results(
            work_id="garden", title="Garden", digest=self.DIGEST,
            passage="A brass key lay under the gate, half-buried in the loam.")
        experiences = self.store.get_experiences()
        self.assertTrue(experiences)
        self.assertTrue(all(e["type"] == "experience" for e in experiences))
        # The experience is concrete enough to be recalled: it keeps an excerpt.
        self.assertIn("brass key", experiences[0]["content"])
        # No durable self-fact/self-preference was created by reading alone.
        self_facts = [m for m in self.store.get_memories("self", status=None)
                      if m["type"] in ("self_fact", "self_preference")]
        self.assertEqual(self_facts, [])

    def test_finishing_a_work_is_a_meaningful_experience(self):
        self.store.record_reading_finish(work_id="garden", title="Garden", chunks=9)
        finished = [e for e in self.store.get_experiences()
                    if e["experience_kind"] == affect.EXPERIENCE_COMPLETED]
        self.assertTrue(finished)
        self.assertIn("Finished reading", finished[0]["content"])

    def test_reading_experiences_move_the_affect_state(self):
        self.store.apply_reading_results(work_id="garden", title="Garden", digest=self.DIGEST)
        state = self.store.current_affect()
        self.assertGreater(state["engagement"], 0.0)
        # The digest carries entities/ideas, so the step is recorded as a
        # discovery rather than a plain read.
        self.assertGreater(state["event_counts"].get("discovered", 0), 0)


# ---------------------------------------------------------------------
# Prompt integration and read-only guarantees
# ---------------------------------------------------------------------
class TestPromptIntegration(_Base):
    def _orchestrator(self):
        return CompanionOrchestrator(self.store, config_dir=self.tmp)

    def test_no_reader_means_no_reading_block(self):
        prompt = self._orchestrator().build_prompt("hello", [])
        self.assertNotIn("ASTRA'S READING", prompt)

    def test_reading_block_appears_when_a_work_is_live(self):
        self.library.add_text("The captain stared at the sea. " * 60,
                              title="Sea Novel", work_id="sea")
        orchestrator = self._orchestrator()
        orchestrator.reader = self.reader()
        prompt = orchestrator.build_prompt("what are you up to?", [])
        self.assertIn("ASTRA'S READING", prompt)
        self.assertIn("Sea Novel", prompt)

    def test_prompt_building_does_not_touch_storage(self):
        self.library.add_text("The captain stared at the sea. " * 60,
                              title="Sea Novel", work_id="sea")
        self.store.apply_reading_results(
            work_id="sea", title="Sea Novel", digest=TestReadingMemories.DIGEST)
        orchestrator = self._orchestrator()
        orchestrator.reader = self.reader()
        before = _hashes(self.tmp)
        orchestrator.build_prompt("tell me about the sea novel", [])
        orchestrator.build_prompt("what are you reading?", [])
        self.assertEqual(before, _hashes(self.tmp))

    def test_reading_diagnostics_surface_why(self):
        self.library.add_text("The captain stared at the sea. " * 60,
                              title="Sea Novel", work_id="sea")
        reader = self.reader()
        reader.note_game_pause("dota2.exe")
        diag = reader.diagnostics()
        self.assertEqual(diag["reason"], reading.IDLE_GAME_RUNNING)
        self.assertEqual(diag["game"], "dota2.exe")
        self.assertEqual(diag["work_id"], "sea")


# ---------------------------------------------------------------------
# The CLI surface: /reading, /library, and the explicit /read escape hatch
# ---------------------------------------------------------------------
class TestCliReadingCommands(_Base):
    def _session(self, model):
        from main import ChatSession

        self.library.add_text("The captain stared at the grey sea. " * 120,
                              title="Sea Novel", work_id="sea")
        orchestrator = CompanionOrchestrator(self.store, config_dir=self.tmp)
        reader = self.reader(model)
        orchestrator.reader = reader
        self.out: list = []
        session = ChatSession(
            orchestrator, None, reader=reader,
            output_fn=self.out.append, input_fn=lambda _: "",
        )
        return session, reader

    def test_read_now_runs_a_cycle_despite_being_interaction(self):
        session, _ = self._session(lambda p: "{}")
        session._handle_slash("/read")
        self.assertTrue(any("Read 1 chunk(s)" in line for line in self.out))
        self.assertGreater(self.library.get_state("sea")["char_offset"], 0)

    def test_read_pause_and_resume_toggle_the_reader(self):
        session, reader = self._session(lambda p: "{}")
        session._handle_slash("/read pause")
        self.assertFalse(reader.enabled)
        session._handle_slash("/read resume")
        self.assertTrue(reader.enabled)

    def test_read_now_still_refuses_while_a_game_runs(self):
        session, reader = self._session(lambda p: "{}")
        reader.note_game_pause("dota2.exe")
        session._handle_slash("/read")
        self.assertTrue(any("game_running" in line for line in self.out))
        self.assertEqual(self.library.get_state("sea")["char_offset"], 0)

    def test_library_lists_ingested_works(self):
        session, _ = self._session(lambda p: "{}")
        session._handle_slash("/library")
        joined = "\n".join(self.out)
        self.assertIn("LIBRARY (1 work(s))", joined)
        self.assertIn("Sea Novel", joined)

    def test_reading_command_reports_status(self):
        session, _ = self._session(lambda p: "{}")
        session._handle_slash("/reading")
        joined = "\n".join(self.out)
        self.assertIn("BACKGROUND READING", joined)
        self.assertIn("Sea Novel", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
