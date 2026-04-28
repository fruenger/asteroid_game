"""Smoke test for b1_game_bridge (needs poliastro / full venv)."""

from __future__ import annotations

import unittest


class TestB1GameBridge(unittest.TestCase):
    def test_session_tick_smoke(self) -> None:
        import b1_game_bridge as b1

        b1.session_init(verbose_scenario=False)
        st = b1.status()
        self.assertTrue(st["ok"])
        self.assertEqual(st["step"], 0)

        out = b1.tick(0.016, 100.0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertTrue(out["ok"])
        self.assertIn("help_overlay_kind", out)
        self.assertEqual(out["help_overlay_kind"], "none")
        self.assertIn("target_guides", out)
        self.assertIsNone(out["target_guides"])
        self.assertIn("sun_x", out)
        self.assertIn("help_body", out)
        self.assertIn("help_story_plain", out)
        self.assertIn("catalog_lines", out)
        self.assertIn("panel_enabled", out)
        self.assertIn("panel_w", out)
        self.assertIn("panel_h", out)
        self.assertEqual(out["step"], 0)
        self.assertEqual(out["catalog_lines"], [])

    def test_align_dome_to_telescope(self) -> None:
        import b1_game_bridge as b1

        b1.session_init(verbose_scenario=False)
        r = b1.align_dome_to_telescope()
        self.assertTrue(r.get("ok"))
        self.assertIn("dome_az_deg", r)

    def test_set_dome_az_deg(self) -> None:
        import b1_game_bridge as b1

        b1.session_init(verbose_scenario=False)
        r = b1.set_dome_az_deg(-37.25)
        self.assertTrue(r.get("ok"))
        self.assertAlmostEqual(float(r["dome_az_deg"]), -37.25, places=4)


if __name__ == "__main__":
    unittest.main()
