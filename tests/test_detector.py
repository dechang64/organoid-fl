# ── tests/test_detector.py ──
"""Tests for the YOLOv11 Detector module.

Note on the class schema
------------------------
This file previously asserted::

    assert det.CLASS_NAMES == ["healthy", "early_stage", "late_stage"]

That assertion pinned a three-class placeholder which matched **no dataset in
the repository** — every real ``data.yaml`` declares four classes. The test was
therefore guarding the bug rather than catching it, and it is now written
against the taxonomy registry so it cannot drift again.
"""

import os
import sys
import unittest
import warnings


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.detector import Detection, OrganoidDetector
from utils.taxonomy import TaxonomyError


class TestDetection(unittest.TestCase):
    def test_creation(self):
        d = Detection(
            bbox=[10, 20, 100, 200],
            class_name="organoid0",
            class_id=0,
            confidence=0.95,
            cx=55, cy=110,
            width=90, height=180,
            area=16200,
        )
        self.assertEqual(d.class_name, "organoid0")
        self.assertEqual(d.confidence, 0.95)
        self.assertEqual(d.area, 16200)

    def test_to_dict(self):
        d = Detection(
            bbox=[0, 0, 50, 50],
            class_name="early_budding",
            class_id=1,
            confidence=0.88,
            cx=25, cy=25,
            width=50, height=50,
            area=2500,
        )
        d_dict = d.to_dict()
        self.assertEqual(d_dict["class_name"], "early_budding")
        self.assertEqual(d_dict["confidence"], 0.88)
        self.assertIn("bbox", d_dict)

    def test_label_and_taxonomy_are_recorded(self):
        d = Detection(
            bbox=[0, 0, 10, 10], class_name="spheroid", class_id=3, confidence=0.5,
            cx=5, cy=5, width=10, height=10, area=100,
            label="Spheroid", taxonomy_name="organoid_4class",
        )
        self.assertEqual(d.to_dict()["taxonomy_name"], "organoid_4class")


class TestOrganoidDetector(unittest.TestCase):
    def test_init(self):
        det = OrganoidDetector(model_size="n")
        self.assertEqual(det.model_size, "n")
        self.assertEqual(det.model_name, "yolo11n.pt")

    def test_class_names_come_from_the_registry(self):
        """Guards against a hard-coded placeholder reappearing."""
        det = OrganoidDetector(model_size="n")
        self.assertEqual(
            det.class_names, ["organoid0", "organoid1", "organoid3", "spheroid"]
        )
        self.assertEqual(det.n_classes, 4)
        self.assertEqual(OrganoidDetector.CLASS_NAMES, det.class_names)

    def test_the_placeholder_schema_is_gone(self):
        det = OrganoidDetector(model_size="n")
        for stale in ("healthy", "early_stage", "late_stage"):
            self.assertNotIn(stale, det.class_names)

    def test_class_list_matches_the_datasets_on_disk(self):
        """The regression that motivated the registry: detector vs data.yaml."""
        import yaml

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "organoid_patches", "data.yaml",
        )
        if not os.path.isfile(path):
            self.skipTest("data/organoid_patches/data.yaml not present")
        with open(path, "r", encoding="utf-8") as fh:
            declared = yaml.safe_load(fh)["names"]
        self.assertEqual(OrganoidDetector().class_names, list(declared))

    def test_every_class_index_resolves(self):
        det = OrganoidDetector()
        for cls_id in range(det.n_classes):
            key, label = det._resolve_class(cls_id)
            self.assertIn(key, det.class_names)
            self.assertIsNotNone(label)

    def test_a_fourth_class_no_longer_becomes_class_3(self):
        """The concrete defect: cls_id=3 used to fall through a 3-element list."""
        det = OrganoidDetector()
        key, _ = det._resolve_class(3)
        self.assertEqual(key, "spheroid")
        self.assertNotEqual(key, "class_3")

    def test_out_of_range_index_warns_by_default(self):
        det = OrganoidDetector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            key, label = det._resolve_class(99)
        self.assertEqual(key, "class_99")
        self.assertIsNone(label)
        self.assertTrue(any("class index 99" in str(w.message) for w in caught))

    def test_out_of_range_index_can_fail_fast(self):
        det = OrganoidDetector(on_unknown_class="raise")
        with self.assertRaises(TaxonomyError):
            det._resolve_class(99)

    def test_passthrough_mode_is_silent(self):
        det = OrganoidDetector(on_unknown_class="passthrough")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            key, _ = det._resolve_class(7)
        self.assertEqual(key, "class_7")
        self.assertEqual(len(caught), 0)

    def test_invalid_unknown_class_policy_rejected(self):
        with self.assertRaises(ValueError):
            OrganoidDetector(on_unknown_class="explode")

    def test_taxonomy_can_be_selected_per_dataset(self):
        det = OrganoidDetector(taxonomy="multiorg_2class")
        self.assertEqual(det.class_names, ["normal", "macros"])


        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "organoid_patches", "data.yaml",
        )
        if os.path.isfile(path):
            det2 = OrganoidDetector(taxonomy=path)
            self.assertEqual(det2.class_names, ["organoid0", "organoid1", "organoid3", "spheroid"])

    def test_count_by_class(self):
        det = OrganoidDetector(model_size="n")
        detections = [
            Detection([0, 0, 50, 50], "organoid0", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0, 0, 50, 50], "organoid0", 0, 0.8, 25, 25, 50, 50, 2500),
            Detection([0, 0, 50, 50], "spheroid", 3, 0.7, 25, 25, 50, 50, 2500),
        ]
        counts = det.count_by_class(detections)
        self.assertEqual(counts, {"organoid0": 2, "spheroid": 1})

    def test_summary_empty(self):
        det = OrganoidDetector(model_size="n")
        summary = det.summary([])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["avg_confidence"], 0)

    def test_summary_nonempty(self):
        det = OrganoidDetector(model_size="n")
        detections = [
            Detection([0, 0, 50, 50], "organoid0", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0, 0, 100, 100], "organoid1", 1, 0.7, 50, 50, 100, 100, 10000),
        ]
        summary = det.summary(detections)
        self.assertEqual(summary["total"], 2)
        self.assertAlmostEqual(summary["avg_confidence"], 0.8)
        self.assertEqual(summary["avg_area"], 6250)
        self.assertEqual(summary["min_area"], 2500)
        self.assertEqual(summary["max_area"], 10000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
