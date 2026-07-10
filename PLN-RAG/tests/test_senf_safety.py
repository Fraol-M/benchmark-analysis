from __future__ import annotations

import unittest
from pathlib import Path

from core.senf import SENFBuilder, BridgeGenerator
from core.senf.models import IdentityEdge


class SENFSafetyTests(unittest.TestCase):
    def test_senf_is_disabled_by_default(self):
        config_text = (
            Path(__file__).resolve().parents[1] / "config.py"
        ).read_text(encoding="utf-8")

        self.assertIn("senf_enabled: bool = False", config_text)
        self.assertIn("senf_emit_bridges: bool = False", config_text)

    def test_builder_keeps_repeated_surface_text_as_separate_mentions(self):
        builder = SENFBuilder()
        senf = builder.build(
            [
                "(: camera_seen_fact (Sees alex camera) (STV 1.0 1.0))",
                "(: camera_owned_fact (Owns sam camera) (STV 1.0 1.0))",
            ],
            text="Alex sees the camera. Sam owns the camera.",
            metadata={"chunk_id": "chunk1"},
        )

        camera_mentions = [
            entity for entity in senf.entities if entity.canonical_text == "camera"
        ]
        self.assertEqual(2, len(camera_mentions))
        self.assertEqual(2, len({entity.id for entity in camera_mentions}))
        self.assertTrue(all(entity.id.startswith("chunk1_f_") for entity in camera_mentions))

    def test_bridge_generator_uses_canonical_surface_terms_when_explicitly_called(self):
        builder = SENFBuilder()
        senf = builder.build(
            [
                "(: camera_seen_fact (Sees Alex camera) (STV 1.0 1.0))",
                "(: camera_owned_fact (Owns Sam camera) (STV 1.0 1.0))",
            ],
            text="Alex sees the camera. Sam owns the camera.",
            metadata={"chunk_id": "chunk1"},
        )
        cameras = [
            entity for entity in senf.entities if entity.canonical_text == "camera"
        ]
        edge = IdentityEdge(
            left=cameras[0].id,
            right=cameras[1].id,
            c_plus=0.15,
            c_minus=0.8,
            reasons_plus=["test"],
        )

        bridges = BridgeGenerator().generate(senf, [edge], [])

        self.assertEqual(1, len(bridges))
        self.assertIn("(SimilarityLink camera camera)", bridges[0])


if __name__ == "__main__":
    unittest.main()
