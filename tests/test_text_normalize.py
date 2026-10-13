"""Text normalization: ingested content is cleaned without changing meaning.

The brief's section 10 asks that memory text stop carrying presentation debris
(markup, entities, ligatures, whitespace) while remaining faithful to what was
actually said. These tests pin both halves: the transforms, and the guarantees
that make them safe to run over the whole corpus - idempotence, code/URL
protection, no-op on clean text, and never writing emptiness over real content.
"""

import os
import tempfile
import unittest

from astra import text_normalize as tn
from astra.memory import TripleMemoryStore


class TestNormalizeTransforms(unittest.TestCase):
    def test_markup_is_stripped(self):
        self.assertEqual(tn.normalize("<p>Hello <b>world</b></p>"), "Hello world")

    def test_entities_are_decoded(self):
        self.assertEqual(tn.normalize("Tom &amp; Jerry &mdash; friends"),
                         "Tom & Jerry \u2014 friends")

    def test_double_escaped_entities_settle(self):
        self.assertEqual(tn.normalize("a &amp;amp; b"), "a & b")

    def test_ligatures_and_fullwidth_fold(self):
        self.assertEqual(tn.normalize("The \ufb01rst \ufb02oor"), "The first floor")
        self.assertEqual(tn.normalize("Ｈｅｌｌｏ"), "Hello")

    def test_invisibles_are_removed(self):
        self.assertEqual(tn.normalize("zero\u200bwidth\u00adsoft"),
                         "zerowidthsoft")

    def test_whitespace_is_collapsed_but_paragraphs_survive(self):
        self.assertEqual(tn.normalize("a   b\tc"), "a b c")
        self.assertEqual(tn.normalize("p1\n\n\n\np2"), "p1\n\np2")

    def test_separator_lines_are_dropped(self):
        self.assertEqual(tn.normalize("***\nChapter 1\n***"), "Chapter 1")

    def test_broken_line_wrap_is_rejoined(self):
        text = ("The quick brown fox jumped over\n"
                "the lazy dog and kept running.")
        self.assertEqual(tn.normalize(text),
                         "The quick brown fox jumped over the lazy dog and kept running.")

    def test_real_breaks_are_kept(self):
        text = "First sentence.\nSecond sentence.\n- item one\n- item two"
        self.assertEqual(tn.normalize(text), text)

    def test_plain_angle_brackets_are_not_treated_as_tags(self):
        self.assertEqual(tn.normalize("if a < b and b > c then done"),
                         "if a < b and b > c then done")


class TestNormalizeSafety(unittest.TestCase):
    def test_urls_are_preserved_verbatim(self):
        text = "See https://example.com/a?b=1&amp;c=2 for more."
        self.assertIn("https://example.com/a?b=1&amp;c=2", tn.normalize(text))

    def test_fenced_code_is_preserved_verbatim(self):
        fence = "```\nblock <tag> &amp; a<b>c\n```"
        self.assertEqual(tn.normalize(fence), fence)

    def test_inline_code_is_preserved_verbatim(self):
        text = "Use `<span> &amp;` in the template."
        self.assertEqual(tn.normalize(text), text)

    def test_clean_text_is_unchanged(self):
        text = "A perfectly ordinary sentence, with punctuation: and a list."
        self.assertEqual(tn.normalize(text), text)
        self.assertFalse(tn.is_noisy(text))

    def test_idempotent_on_a_corpus_of_mixed_input(self):
        samples = [
            "<p>Hello&nbsp;<b>world</b> &amp; goodbye</p>",
            "The \ufb01rst \ufb02oor  was  wide.",
            "***\nChapter 1\n***",
            "a   b\tc",
            "See https://example.com/a?b=1&amp;c=2 for more.",
            "```\ncode <x> &amp;\n```",
            "Text with \u200bzero width\u00ad soft hyphen.",
            "",
            "   ",
        ]
        for s in samples:
            once = tn.normalize(s)
            self.assertEqual(tn.normalize(once), once, s)

    def test_markup_only_input_is_reported_as_emptied(self):
        view = tn.diff("<p></p>")
        self.assertTrue(view["emptied"])
        self.assertEqual(view["after"], "")


class TestStoreNormalization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TripleMemoryStore(data_dir=self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_content_is_normalized_at_the_write_choke_point(self):
        mem_id = self.store.add_memory(
            "roum", "<p>Likes <b>tea</b> &amp; biscuits</p>", "explicit_fact", "user_statement")
        mem = self.store.get_memory("roum", mem_id)
        self.assertEqual(mem["content"], "Likes tea & biscuits")

    def test_preview_does_not_write(self):
        mem_id = self.store.add_memory(
            "roum", "clean text", "explicit_fact", "user_statement")
        # Force noise in directly, bypassing the write path, to model legacy data.
        self.store.memories["roum"][0]["content"] = "<p>noisy &amp; text</p>"
        before = self.store.get_memory("roum", mem_id)["content"]
        preview = self.store.preview_text_normalization()
        self.assertEqual(preview["changed"], 1)
        self.assertEqual(self.store.get_memory("roum", mem_id)["content"], before)

    def test_apply_rewrites_content_and_is_idempotent(self):
        self.store.add_memory("roum", "clean", "explicit_fact", "user_statement")
        self.store.memories["roum"][0]["content"] = "<p>noisy &amp; text</p>"
        first = self.store.normalize_memory_texts()
        self.assertEqual(first["changed"], 1)
        self.assertEqual(self.store.memories["roum"][0]["content"], "noisy & text")
        second = self.store.normalize_memory_texts()
        self.assertEqual(second.get("changed", 0), 0)

    def test_identity_and_authority_fields_are_untouched(self):
        # Normalization rewrites *content only*. Every provenance/authority/
        # identity field must be byte-identical before and after.
        mem_id = self.store.add_memory(
            "roum", "Likes tea", "explicit_fact", "user_statement",
            confidence=0.9)
        self.store.memories["roum"][0]["content"] = "<p>Likes <b>tea</b></p>"
        before = dict(self.store.get_memory("roum", mem_id))
        self.store.normalize_memory_texts()
        after = self.store.get_memory("roum", mem_id)
        self.assertEqual(after["content"], "Likes tea")
        for field in ("id", "type", "source", "confidence", "timestamp",
                      "authority_source", "status"):
            self.assertEqual(after.get(field), before.get(field), field)

    def test_markup_only_content_is_never_written_empty(self):
        self.store.add_memory("roum", "keep me", "explicit_fact", "user_statement")
        self.store.memories["roum"][0]["content"] = "<p></p>"
        self.store.normalize_memory_texts()
        self.assertEqual(self.store.memories["roum"][0]["content"], "<p></p>")

    def test_audit_file_is_written(self):
        self.store.add_memory("roum", "clean", "explicit_fact", "user_statement")
        self.store.memories["roum"][0]["content"] = "a &amp; b"
        self.store.normalize_memory_texts()
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "text_normalization_audit.json")))


if __name__ == "__main__":
    unittest.main()
