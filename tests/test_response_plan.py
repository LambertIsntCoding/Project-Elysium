"""Tests for response planning and the voice-renderer separation."""

import unittest

from astra import response_plan as rp


class TestPlanning(unittest.TestCase):
    def test_error_is_marked_as_error(self):
        plan = rp.plan_response("hi", "[Error communicating...]", error=True)
        self.assertEqual(plan.function, rp.FUNC_ERROR)
        self.assertEqual(plan.interruption_suitability, 0.0)

    def test_proactive_plan_is_initiative(self):
        plan = rp.plan_response("", "I was thinking about the lighthouse.",
                                proactive=True)
        self.assertEqual(plan.function, rp.FUNC_INITIATIVE)
        self.assertTrue(plan.proactive)

    def test_plan_is_deterministic(self):
        a = rp.plan_response("why is the sky blue?", "Rayleigh scattering.")
        b = rp.plan_response("why is the sky blue?", "Rayleigh scattering.")
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_plan_carries_no_personality_content(self):
        plan = rp.plan_response("hello", "hey!")
        data = plan.to_dict()
        # The plan describes delivery, not identity or memory.
        for forbidden in ("memory", "personality", "identity", "trait"):
            self.assertNotIn(forbidden, data)


class TestVoiceRenderer(unittest.TestCase):
    def test_default_renderer_is_a_pass_through(self):
        renderer = rp.VoiceRenderer()
        plan = rp.plan_response("hi", "hello there")
        self.assertEqual(renderer.render("hello there", plan), "hello there")

    def test_null_renderer_is_unavailable(self):
        self.assertFalse(rp.NullVoiceRenderer().available())

    def test_recording_renderer_does_not_change_text(self):
        renderer = rp.RecordingVoiceRenderer()
        plan = rp.plan_response("hi", "same text")
        out = renderer.render("same text", plan)
        self.assertEqual(out, "same text")
        self.assertEqual(renderer.calls[0]["text"], "same text")
        self.assertEqual(renderer.calls[0]["plan"]["function"], plan.function)


if __name__ == "__main__":
    unittest.main(verbosity=2)
