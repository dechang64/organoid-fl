# ── tests/test_morphometrics.py ──
"""Tests for calibrated, per-object morphology measurement.

The central regression guard here is
:meth:`TestPerObjectScope.test_merged_mask_no_longer_produces_impossible_ratios`.
The previous implementation mixed two scopes inside one ratio — a global pixel
count over a single-contour convex hull — and could report ``solidity = 5.18``
for a two-object mask. That is the mechanism behind the ``solidity`` AUC of
0.272 (below the 0.5 of random) recorded in ``docs/experiment_tracker.md:149``.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.morphometrics import (
    MorphologyConfig,
    label_map,
    mask_quality,
    mask_to_instances,
    measure_mask,
)
from analysis.schema import Provenance, ReviewStatus

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def disk(shape=(128, 128), cx=64, cy=64, r=10):
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    return mask


def two_disks(shape=(200, 200)):
    if cv2 is None:
        raise unittest.SkipTest("cv2 required for the two-disk fixture")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (60, 60), 20, 1, -1)
    cv2.circle(mask, (150, 150), 10, 1, -1)
    return mask


class TestLabelMap(unittest.TestCase):
    def test_empty_mask(self):
        labels, n = label_map(np.zeros((10, 10), dtype=bool))
        self.assertEqual(n, 0)
        self.assertEqual(labels.sum(), 0)

    def test_counts_components(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:5, 2:5] = True
        mask[10:13, 10:13] = True
        labels, n = label_map(mask)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(set(labels.ravel().tolist()) - {0}), [1, 2])

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            label_map(np.zeros((3, 4, 5), dtype=bool))


class TestMaskQuality(unittest.TestCase):
    def test_single_object_mask(self):
        q = mask_quality(disk())
        self.assertEqual(q["n_components"], 1)
        self.assertFalse(q["is_multi_component"])
        self.assertAlmostEqual(q["largest_component_fraction"], 1.0, places=6)

    def test_merged_mask_is_flagged(self):
        q = mask_quality(two_disks())
        self.assertEqual(q["n_components"], 2)
        self.assertTrue(q["is_multi_component"])
        self.assertLess(q["largest_component_fraction"], 1.0)

    def test_empty_mask_has_zero_components(self):
        q = mask_quality(np.zeros((10, 10), dtype=bool))
        self.assertEqual(q["n_components"], 0)
        self.assertEqual(q["largest_component_fraction"], 0.0)


class TestPerObjectScope(unittest.TestCase):
    def test_merged_mask_no_longer_produces_impossible_ratios(self):
        mask = two_disks()
        measurements = measure_mask(mask, pixel_size_um=0.325, config=MorphologyConfig(min_area_px=4))

        self.assertEqual(len(measurements), 2, "each object must be measured separately")

        # Both ratios are bounded by construction for every object.
        for m in measurements:
            self.assertIsNotNone(m.solidity)
            self.assertLessEqual(m.solidity, 1.0)
            self.assertGreaterEqual(m.solidity, 0.0)
            self.assertIsNotNone(m.circularity)
            self.assertLessEqual(m.circularity, 1.05)
            self.assertGreaterEqual(m.circularity, 0.0)

        # And the old defect is explicitly documented here.
        global_area = int(mask.sum())
        component_areas = sum(m.area_px for m in measurements)
        self.assertLessEqual(component_areas, global_area)

    def test_reproduces_the_old_defect_for_the_record(self):
        """Show that the superseded formula really was out of range.

        This keeps the diagnosis in the test suite instead of only in a commit
        message, so the reason for the change cannot be lost.
        """
        if cv2 is None:
            self.skipTest("cv2 required")
        mask = two_disks()
        global_area = int(mask.sum())
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        old_hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        old_solidity = global_area / old_hull_area if old_hull_area else 0.0

        self.assertGreater(
            old_solidity, 1.0,
            "the superseded global-area / local-hull ratio exceeded 1, which is "
            "impossible for a shape descriptor",
        )

    def test_measurements_are_ordered_largest_first(self):
        measurements = measure_mask(two_disks(), config=MorphologyConfig(min_area_px=4))
        areas = [m.area_px for m in measurements]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_each_measurement_knows_its_mask_held_several_objects(self):
        measurements = measure_mask(two_disks(), config=MorphologyConfig(min_area_px=4))
        for m in measurements:
            self.assertTrue(m.is_multi_component)
            self.assertEqual(m.n_components, 2)
            self.assertFalse(m.is_single_object)


class TestPhysicalUnits(unittest.TestCase):
    def test_calibrated_geometry(self):
        mask = disk(r=10)
        m = measure_mask(mask, pixel_size_um=0.5, config=MorphologyConfig(min_area_px=1))[0]
        self.assertTrue(m.is_physically_calibrated)
        self.assertAlmostEqual(m.area_um2, m.area_px * 0.25, places=4)
        self.assertAlmostEqual(m.centroid_um[0], m.centroid_px[0] * 0.5, places=6)

    def test_uncalibrated_geometry_stays_none(self):
        """Pixel counts must never be reported under a ``_um`` name."""
        m = measure_mask(disk(r=10), config=MorphologyConfig(min_area_px=1))[0]
        self.assertFalse(m.is_physically_calibrated)
        self.assertIsNone(m.area_um2)
        self.assertIsNone(m.perimeter_um)
        self.assertIsNone(m.equivalent_diameter_um)
        self.assertIsNone(m.centroid_um)
        self.assertGreater(m.area_px, 0)  # pixel values still available

    def test_equivalent_diameter_comes_from_area(self):
        mask = disk(r=10)
        m = measure_mask(mask, pixel_size_um=1.0, config=MorphologyConfig(min_area_px=1))[0]
        expected_px = 2.0 * np.sqrt(m.area_px / np.pi)
        self.assertAlmostEqual(m.equivalent_diameter_um, expected_px, places=4)

    def test_larger_object_has_larger_area(self):
        small = measure_mask(disk(r=6), pixel_size_um=0.5, config=MorphologyConfig(min_area_px=1))[0]
        large = measure_mask(disk(r=14), pixel_size_um=0.5, config=MorphologyConfig(min_area_px=1))[0]
        self.assertLess(small.area_um2, large.area_um2)
        self.assertLess(small.equivalent_diameter_um, large.equivalent_diameter_um)


class TestShapeDescriptors(unittest.TestCase):
    def test_circle_has_solidity_near_one(self):
        m = measure_mask(disk(r=12), config=MorphologyConfig(min_area_px=1))[0]
        self.assertGreater(m.solidity, 0.98)
        self.assertLess(m.eccentricity, 0.3)

    def test_concave_shape_has_solidity_below_one(self):
        """A C-shape is the classic case where solidity carries signal."""
        if cv2 is None:
            self.skipTest("cv2 required")
        mask = np.zeros((80, 80), np.uint8)
        cv2.circle(mask, (40, 40), 25, 1, -1)
        cv2.circle(mask, (40, 40), 14, 0, -1)
        mask[38:52, 40:70] = 0  # open the ring into a C

        m = measure_mask(mask, config=MorphologyConfig(min_area_px=1))[0]
        self.assertLess(m.solidity, 0.95)
        self.assertGreater(m.solidity, 0.0)

    def test_hole_is_detected(self):
        if cv2 is None:
            self.skipTest("cv2 required")
        mask = np.zeros((60, 60), np.uint8)
        cv2.circle(mask, (30, 30), 20, 1, -1)
        cv2.circle(mask, (30, 30), 8, 0, -1)

        m = measure_mask(mask, config=MorphologyConfig(min_area_px=1))[0]
        self.assertTrue(m.has_holes)
        self.assertEqual(m.euler_number, 0)  # one component, one hole

    def test_elongated_shape_has_aspect_ratio_above_one(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[45:55, 10:90] = 1
        m = measure_mask(mask, config=MorphologyConfig(min_area_px=1))[0]
        self.assertGreater(m.aspect_ratio, 3.0)
        self.assertGreater(m.eccentricity, 0.9)


class TestBorderAndSizeGates(unittest.TestCase):
    def test_touching_border_is_flagged(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[0:20, 20:40] = 1
        m = measure_mask(mask, config=MorphologyConfig(min_area_px=1))[0]
        self.assertTrue(m.touches_border)

    def test_interior_object_is_not_flagged(self):
        m = measure_mask(disk(r=8, cx=32, cy=32), config=MorphologyConfig(min_area_px=1))[0]
        self.assertFalse(m.touches_border)

    def test_border_margin_widens_the_gate(self):
        mask = np.zeros((64, 64), np.uint8)
        mask[4:20, 20:40] = 1
        strict = measure_mask(mask, config=MorphologyConfig(min_area_px=1))[0]
        relaxed = measure_mask(mask, config=MorphologyConfig(min_area_px=1, border_margin_px=8))[0]
        self.assertFalse(strict.touches_border)
        self.assertTrue(relaxed.touches_border)

    def test_min_area_drops_specks(self):
        mask = disk(r=10)
        mask[0:2, 0:2] = 1  # a 4-pixel speck
        self.assertEqual(len(measure_mask(mask, config=MorphologyConfig(min_area_px=8))), 1)
        self.assertEqual(len(measure_mask(mask, config=MorphologyConfig(min_area_px=1))), 2)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            MorphologyConfig(connectivity=6)
        with self.assertRaises(ValueError):
            MorphologyConfig(min_area_px=-1)


class TestIntensity(unittest.TestCase):
    def test_intensity_is_measured_on_the_native_scale(self):
        """Passing 16-bit data must not be silently rescaled to 0-255."""
        mask = disk(r=10)
        intensity = np.random.RandomState(0).randint(1000, 60000, mask.shape).astype(np.uint16)
        m = measure_mask(
            mask, intensity=intensity, channel_names=["GFP"],
            config=MorphologyConfig(min_area_px=1),
        )[0]
        inside = intensity[mask.astype(bool)]
        self.assertAlmostEqual(m.mean_intensity["GFP"], float(inside.mean()), places=3)
        self.assertAlmostEqual(m.integral_intensity["GFP"], float(inside.sum()), places=1)
        self.assertGreater(m.mean_intensity["GFP"], 255.0)

    def test_multichannel_intensity_is_kept_separate(self):
        mask = disk(r=10)
        intensity = np.stack([
            np.full(mask.shape, 100, np.uint16),
            np.full(mask.shape, 5000, np.uint16),
        ])
        m = measure_mask(
            mask, intensity=intensity, channel_names=["DAPI", "GFP"],
            config=MorphologyConfig(min_area_px=1),
        )[0]
        self.assertEqual(set(m.mean_intensity), {"DAPI", "GFP"})
        self.assertNotAlmostEqual(m.mean_intensity["DAPI"], m.mean_intensity["GFP"])

    def test_channel_name_count_mismatch_raises(self):
        mask = disk(r=10)
        intensity = np.stack([np.zeros(mask.shape, np.uint16)] * 3)
        with self.assertRaises(ValueError):
            measure_mask(mask, intensity=intensity, channel_names=["A", "B"])

    def test_intensity_can_be_disabled(self):
        mask = disk(r=10)
        m = measure_mask(
            mask, intensity=np.full(mask.shape, 500, np.uint16),
            config=MorphologyConfig(min_area_px=1, compute_intensity=False),
        )[0]
        self.assertEqual(dict(m.mean_intensity), {})


class TestMaskToInstances(unittest.TestCase):
    def test_splits_a_merged_mask_into_one_instance_per_object(self):
        instances = mask_to_instances(
            two_disks(),
            image_id="img1",
            class_key="spheroid",
            taxonomy_name="organoid_4class",
            pixel_size_um=0.5,
            config=MorphologyConfig(min_area_px=4),
        )
        self.assertEqual(len(instances), 2)
        self.assertEqual(len({i.id for i in instances}), 2)
        for inst in instances:
            self.assertEqual(inst.image_id, "img1")
            self.assertEqual(inst.class_key, "spheroid")
            self.assertTrue(inst.has_mask)
            self.assertIsNotNone(inst.measurement)
            self.assertIsNotNone(inst.bbox_um)

    def test_provenance_and_review_default_to_preannotation(self):
        inst = mask_to_instances(
            disk(r=8), image_id="img1", class_key="organoid0",
            taxonomy_name="organoid_4class",
        )[0]
        self.assertIs(inst.provenance, Provenance.MODEL_PREANNOTATION)
        self.assertIs(inst.review_status, ReviewStatus.PRE_ANNOTATED)
        self.assertFalse(inst.is_ground_truth)

    def test_explicit_provenance_is_honoured(self):
        inst = mask_to_instances(
            disk(r=8), image_id="img1", class_key="organoid0",
            taxonomy_name="organoid_4class",
            provenance=Provenance.HUMAN, review_status=ReviewStatus.ADJUDICATED,
        )[0]
        self.assertTrue(inst.is_ground_truth)

    def test_merged_mask_leaves_an_audit_note(self):
        inst = mask_to_instances(
            two_disks(), image_id="img1", class_key="organoid0",
            taxonomy_name="organoid_4class", config=MorphologyConfig(min_area_px=4),
        )[0]
        self.assertIsNotNone(inst.notes)
        self.assertIn("mask_quality", inst.notes)

    def test_keep_masks_false_drops_the_array_but_keeps_geometry(self):
        inst = mask_to_instances(
            disk(r=8), image_id="img1", class_key="organoid0",
            taxonomy_name="organoid_4class", keep_masks=False,
        )[0]
        self.assertFalse(inst.has_mask)
        self.assertGreater(inst.measurement.area_px, 0)

    def test_empty_mask_yields_no_instances(self):
        self.assertEqual(
            mask_to_instances(
                np.zeros((16, 16), bool), image_id="i", class_key="c", taxonomy_name="t"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
