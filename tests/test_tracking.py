# ── tests/test_tracking.py ──
"""Tests for time-lapse instance tracking."""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.morphometrics import mask_to_instances
from analysis.schema import Instance
from analysis.tracking import (
    TrackingConfig,
    centroid_distance,
    iou,
    link_tracks,
)

PIXEL_SIZE_UM = 0.5


def blob(cx, cy, r=8, shape=(160, 160)):
    mask = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    return mask.astype(bool)


def make_instance(cx, cy, r=8, t=0, keep_mask=True):
    return mask_to_instances(
        blob(cx, cy, r),
        image_id="img",
        class_key="organoid0",
        taxonomy_name="organoid_4class",
        pixel_size_um=PIXEL_SIZE_UM,
        t_index=t,
        keep_masks=keep_mask,
    )


def strip_mask(instance):
    return instance.with_mask(None)


class TestGeometryHelpers(unittest.TestCase):
    def test_iou_of_identical_masks_is_one(self):
        a = make_instance(40, 40)[0]
        self.assertAlmostEqual(iou(a, a), 1.0, places=6)

    def test_iou_of_disjoint_masks_is_zero(self):
        a = make_instance(20, 20)[0]
        b = make_instance(120, 120)[0]
        self.assertEqual(iou(a, b), 0.0)

    def test_iou_returns_none_without_masks(self):
        """'Cannot compute' must be distinguishable from 'no overlap'."""
        a = strip_mask(make_instance(20, 20)[0])
        b = strip_mask(make_instance(20, 20)[0])
        self.assertIsNone(iou(a, b))

    def test_iou_returns_none_for_mismatched_shapes(self):
        big = make_instance(50, 50, r=8)[0]
        small = mask_to_instances(
            blob(50, 50, r=8, shape=(80, 80)), image_id="i", class_key="c",
            taxonomy_name="t", pixel_size_um=PIXEL_SIZE_UM,
        )[0]
        self.assertIsNone(iou(big, small))

    def test_centroid_distance_uses_physical_scale(self):
        a = make_instance(20, 40)[0]
        b = make_instance(40, 40)[0]
        # 20 px apart at 0.5 µm/px
        self.assertAlmostEqual(centroid_distance(a, b, pixel_size_um=0.5), 10.0, places=3)
        self.assertAlmostEqual(centroid_distance(a, b), 20.0, places=3)

    def test_centroid_distance_falls_back_to_bbox(self):
        a = strip_mask(make_instance(20, 40)[0])
        b = strip_mask(make_instance(40, 40)[0])
        self.assertLess(centroid_distance(a, b, pixel_size_um=0.5), np.inf)

    def test_centroid_distance_is_infinite_without_any_geometry(self):
        bare_a = Instance(id="a", image_id="i", class_key="c", taxonomy_name="t")
        bare_b = Instance(id="b", image_id="i", class_key="c", taxonomy_name="t")
        self.assertEqual(centroid_distance(bare_a, bare_b), float("inf"))


class TestLinking(unittest.TestCase):
    def test_empty_input(self):
        result = link_tracks({})
        self.assertEqual(result.n_tracks, 0)
        self.assertEqual(result.instances, [])

    def test_single_frame_creates_one_track(self):
        result = link_tracks({0: make_instance(40, 40)}, pixel_size_um=PIXEL_SIZE_UM)
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.tracks[0].length, 1)
        self.assertTrue(result.tracks[0].is_root)

    def test_stationary_object_keeps_one_identity(self):
        frames = {t: make_instance(50, 50, t=t) for t in range(4)}
        result = link_tracks(frames, pixel_size_um=PIXEL_SIZE_UM)
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.tracks[0].length, 4)
        self.assertEqual(result.tracks[0].start_t, 0)
        self.assertEqual(result.tracks[0].end_t, 3)

    def test_two_objects_are_not_merged(self):
        frames = {
            t: make_instance(20 + 4 * t, 30, t=t) + make_instance(130 - 4 * t, 130, t=t)
            for t in range(4)
        }
        result = link_tracks(frames, pixel_size_um=PIXEL_SIZE_UM)
        self.assertEqual(result.n_tracks, 2)
        self.assertTrue(all(tr.length == 4 for tr in result.tracks))

    def test_moving_object_survives_a_missed_detection(self):
        """Gap tolerance: the track must bridge one absent frame."""
        positions = [20, 26, 32, 38, 44]
        frames = {t: make_instance(cx, 80, t=t) for t, cx in enumerate(positions)}
        del frames[2]

        result = link_tracks(
            frames,
            config=TrackingConfig(max_centroid_dist_um=8.0, max_frame_gap=1, min_iou=0.1),
            pixel_size_um=PIXEL_SIZE_UM,
        )
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.tracks[0].length, 4)
        self.assertEqual(result.tracks[0].start_t, 0)
        self.assertEqual(result.tracks[0].end_t, 4)

    def test_too_large_a_jump_is_not_linked(self):
        frames = {0: make_instance(20, 80), 1: make_instance(140, 80, t=1)}
        result = link_tracks(
            frames, config=TrackingConfig(max_centroid_dist_um=5.0, min_iou=0.5),
            pixel_size_um=PIXEL_SIZE_UM,
        )
        self.assertEqual(result.n_tracks, 2, "a distant object must start a new track")

    def test_objects_on_different_z_planes_are_not_linked(self):
        from dataclasses import replace

        frame0 = make_instance(50, 50)
        frame1 = [replace(i, z_index=3) for i in make_instance(50, 50, t=1)]
        result = link_tracks({0: frame0, 1: frame1}, pixel_size_um=PIXEL_SIZE_UM)
        self.assertEqual(result.n_tracks, 2, "a Z jump must not be read as motion")

    def test_distance_only_fallback_without_masks(self):
        frames = {t: [strip_mask(i) for i in make_instance(20 + 12 * t, 80, t=t)] for t in range(4)}
        result = link_tracks(
            frames,
            config=TrackingConfig(max_centroid_dist_um=10.0, min_iou=0.1),
            pixel_size_um=PIXEL_SIZE_UM,
        )
        self.assertEqual(result.n_tracks, 1)
        self.assertEqual(result.tracks[0].length, 4)

    def test_pixel_space_gate_when_no_scale_is_known(self):
        frames = {t: [strip_mask(i) for i in make_instance(20 + 8 * t, 80, t=t)] for t in range(4)}
        result = link_tracks(
            frames, config=TrackingConfig(max_centroid_dist_px=12.0, min_iou=0.1)
        )
        self.assertEqual(result.n_tracks, 1)

    def test_warns_when_the_distance_gate_is_silently_unusable(self):
        """A µm threshold with no scale is a footgun; say so."""
        frames = {0: [strip_mask(i) for i in make_instance(20, 80)], 1: [strip_mask(i) for i in make_instance(26, 80, t=1)]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            link_tracks(frames, config=TrackingConfig(max_centroid_dist_um=5.0, min_iou=0.1))
        self.assertTrue(
            any("distance gating is disabled" in str(w.message) for w in caught),
            "expected a warning explaining that the µm gate cannot be applied",
        )


class TestDivision(unittest.TestCase):
    def test_one_parent_two_daughters_creates_lineage(self):
        frames = {
            0: make_instance(80, 80, r=10),
            1: make_instance(74, 80, r=6, t=1) + make_instance(87, 80, r=6, t=1),
        }
        result = link_tracks(
            frames,
            config=TrackingConfig(max_centroid_dist_um=12.0, min_iou=0.05),
            pixel_size_um=PIXEL_SIZE_UM,
        )

        self.assertEqual(result.n_tracks, 3, "parent plus two daughters")
        self.assertEqual(result.n_divisions, 2)

        parent = next(tr for tr in result.tracks if tr.is_root)
        self.assertTrue(parent.is_division)
        self.assertEqual(parent.division_events, [1])
        self.assertEqual(len(parent.children), 2)
        # The parent does not continue past the division.
        self.assertEqual(parent.end_t, 0)
        self.assertEqual(parent.length, 1)

        daughters = [tr for tr in result.tracks if tr.lineage_parent is not None]
        self.assertEqual(len(daughters), 2)
        self.assertTrue(all(tr.lineage_parent == parent.id for tr in daughters))
        self.assertTrue(all(tr.start_t == 1 for tr in daughters))

    def test_an_equal_size_split_is_treated_as_a_matching_error(self):
        """Two daughters each as large as the parent is not biology."""
        frames = {
            0: make_instance(80, 80, r=10),
            1: make_instance(73, 80, r=10, t=1) + make_instance(88, 80, r=10, t=1),
        }
        result = link_tracks(
            frames,
            config=TrackingConfig(max_centroid_dist_um=12.0, min_iou=0.05),
            pixel_size_um=PIXEL_SIZE_UM,
        )
        self.assertEqual(result.n_divisions, 0)

    def test_division_detection_can_be_disabled(self):
        """With divisions off, the parent simply continues and the extra
        object is an unrelated new track — no lineage is invented."""
        frames = {
            0: make_instance(80, 80, r=10),
            1: make_instance(74, 80, r=6, t=1) + make_instance(87, 80, r=6, t=1),
        }
        result = link_tracks(
            frames,
            config=TrackingConfig(max_centroid_dist_um=12.0, min_iou=0.05, allow_division=False),
            pixel_size_um=PIXEL_SIZE_UM,
        )
        self.assertEqual(result.n_divisions, 0)
        self.assertEqual(result.n_tracks, 2)
        self.assertTrue(all(tr.lineage_parent is None for tr in result.tracks))


class TestResultContract(unittest.TestCase):
    def setUp(self):
        frames = {t: make_instance(20 + 6 * t, 80, t=t) for t in range(4)}
        self.result = link_tracks(frames, pixel_size_um=PIXEL_SIZE_UM)

    def test_every_instance_is_returned_with_its_track_id(self):
        self.assertEqual(len(self.result.instances), 4)
        for inst in self.result.instances:
            self.assertIsNotNone(inst.track_id)
            self.assertEqual(self.result.assignments[inst.id], inst.track_id)

    def test_instance_records_are_consistent_with_the_assignments(self):
        for inst in self.result.instances:
            self.assertEqual(self.result.assignments[inst.id], inst.track_id)

    def test_track_ids_are_stable_across_reruns_with_the_same_input(self):
        """The same instance must always land in a track with the same id."""
        from dataclasses import replace as _replace

        first_frames = {t: make_instance(20 + 6 * t, 80, t=t) for t in range(4)}
        first = link_tracks(first_frames, pixel_size_um=PIXEL_SIZE_UM)

        # Re-run with instances that already carry track ids: identity is sticky.
        carried = {
            t: [_replace(i, track_id=first.assignments[i.id]) for i in first_frames[t]]
            for t in first_frames
        }
        before = {i.id: i.track_id for i in carried[0]}
        second = link_tracks(carried, pixel_size_um=PIXEL_SIZE_UM)
        after = {i.id: i.track_id for i in second.instances if i.id in before}
        self.assertEqual(before, after)

    def test_inputs_are_not_mutated(self):
        frames = {t: make_instance(20, 80, t=t) for t in range(2)}
        original = frames[0][0]
        self.assertIsNone(original.track_id)
        link_tracks(frames, pixel_size_um=PIXEL_SIZE_UM)
        self.assertIsNone(original.track_id)

    def test_serialisation(self):
        payload = self.result.to_dict()
        self.assertEqual(payload["n_tracks"], 1)
        self.assertEqual(payload["n_divisions"], 0)
        self.assertAlmostEqual(payload["mean_length"], 4.0)
        self.assertEqual(len(payload["tracks"]), 1)

    def test_mean_length_of_empty_result(self):
        self.assertEqual(link_tracks({}).mean_length, 0.0)


class TestConfigValidation(unittest.TestCase):
    def test_invalid_iou_bounds(self):
        with self.assertRaises(ValueError):
            TrackingConfig(min_iou=1.5)
        with self.assertRaises(ValueError):
            TrackingConfig(min_iou=-0.1)

    def test_negative_gap_rejected(self):
        with self.assertRaises(ValueError):
            TrackingConfig(max_frame_gap=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
