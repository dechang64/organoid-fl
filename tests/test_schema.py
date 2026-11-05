# ── tests/test_schema.py ──
"""Tests for the organoid-native data model."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.schema import (
    ChannelSpec,
    Image,
    Instance,
    Measurement,
    Provenance,
    ReviewStatus,
    Specimen,
    Track,
    new_id,
)


class TestProvenanceAndReview(unittest.TestCase):
    def test_only_human_consensus_imported_count_as_ground_truth(self):
        self.assertTrue(Provenance.HUMAN.is_ground_truth)
        self.assertTrue(Provenance.CONSENSUS.is_ground_truth)
        self.assertTrue(Provenance.IMPORTED.is_ground_truth)
        # A model's own pre-annotation must never evaluate that model.
        self.assertFalse(Provenance.MODEL_PREANNOTATION.is_ground_truth)
        self.assertFalse(Provenance.PROPAGATED.is_ground_truth)
        self.assertFalse(Provenance.SYNTHETIC.is_ground_truth)

    def test_review_signoff_states(self):
        self.assertTrue(ReviewStatus.REVIEWED.is_signoff)
        self.assertTrue(ReviewStatus.ADJUDICATED.is_signoff)
        self.assertFalse(ReviewStatus.PRE_ANNOTATED.is_signoff)
        self.assertFalse(ReviewStatus.UNREVIEWED.is_signoff)

    def test_instance_requires_both_provenance_and_review_for_ground_truth(self):
        base = dict(id="i1", image_id="img1", class_key="spheroid", taxonomy_name="organoid_4class")
        reviewed_model_output = Instance(
            **base, provenance=Provenance.MODEL_PREANNOTATION, review_status=ReviewStatus.REVIEWED
        )
        self.assertFalse(reviewed_model_output.is_ground_truth)

        unreviewed_human = Instance(
            **base, provenance=Provenance.HUMAN, review_status=ReviewStatus.ANNOTATED
        )
        self.assertFalse(unreviewed_human.is_ground_truth)

        gold = Instance(
            **base, provenance=Provenance.HUMAN, review_status=ReviewStatus.ADJUDICATED
        )
        self.assertTrue(gold.is_ground_truth)

    # ── PRD 7.4: assisted decisions ──────────────────────────────────────

    def test_model_selected_is_never_ground_truth(self):
        """A label the model chose cannot be evidence about that model."""
        self.assertFalse(Provenance.MODEL_SELECTED.is_ground_truth)

    def test_assisted_human_label_stays_human_and_countable(self):
        """PRD 7.4: provenance HUMAN, assisted_by model id (never CONSENSUS)."""
        instance = Instance(
            id="i1",
            image_id="img1",
            class_key="spheroid",
            taxonomy_name="organoid_4class",
            provenance=Provenance.HUMAN,
            review_status=ReviewStatus.ADJUDICATED,
            assisted_by="organoid-seg-0.9.3",
            model_rank_agreement=True,
        )
        self.assertTrue(instance.is_ground_truth)
        self.assertEqual(instance.to_dict()["assisted_by"], "organoid-seg-0.9.3")
        self.assertTrue(instance.to_dict()["model_rank_agreement"])

    def test_assistance_cannot_be_recorded_as_consensus(self):
        with self.assertRaises(ValueError):
            Instance(
                id="i1",
                image_id="img1",
                class_key="spheroid",
                taxonomy_name="organoid_4class",
                provenance=Provenance.CONSENSUS,
                assisted_by="organoid-seg-0.9.3",
            )

    def test_blank_assisted_by_is_rejected(self):
        with self.assertRaises(ValueError):
            Instance(
                id="i1",
                image_id="img1",
                class_key="spheroid",
                taxonomy_name="organoid_4class",
                assisted_by="   ",
            )


class TestChannelSpec(unittest.TestCase):
    def test_valid_and_invalid(self):
        ch = ChannelSpec(name="GFP", index=1, stain="GFP", wavelength_nm=488.0)
        self.assertEqual(ch.to_dict()["stain"], "GFP")
        self.assertEqual(ChannelSpec.from_dict(ch.to_dict()), ch)
        with self.assertRaises(ValueError):
            ChannelSpec(name="", index=0)
        with self.assertRaises(ValueError):
            ChannelSpec(name="A", index=-1)


class TestImage(unittest.TestCase):
    def _image(self, **kw):
        defaults = dict(id="img1", path="/tmp/a.tiff", width_px=2048, height_px=1024)
        defaults.update(kw)
        return Image(**defaults)

    def test_calibration_derived_extent(self):
        img = self._image(pixel_size_um=0.325)
        self.assertTrue(img.is_calibrated)
        w, h = img.extent_um
        self.assertAlmostEqual(w, 2048 * 0.325, places=6)
        self.assertAlmostEqual(h, 1024 * 0.325, places=6)

    def test_uncalibrated_image_reports_no_extent(self):
        img = self._image()
        self.assertFalse(img.is_calibrated)
        self.assertIsNone(img.extent_um)

    def test_dimensionality_flags(self):
        self.assertFalse(self._image().is_3d)
        self.assertFalse(self._image().is_timelapse)
        self.assertTrue(self._image(n_z=5).is_3d)
        self.assertTrue(self._image(n_t=10, time_interval_min=15).is_timelapse)
        self.assertTrue(self._image(n_z=5, n_t=10).is_4d)

    def test_z_extent_and_duration(self):
        img = self._image(n_z=11, z_step_um=1.5, n_t=9, time_interval_min=20.0)
        self.assertAlmostEqual(img.z_extent_um, 15.0)
        self.assertAlmostEqual(img.duration_min, 160.0)

    def test_channel_lookup_by_name_and_stain(self):
        img = self._image(channels=(
            ChannelSpec(name="C0", index=0, stain="DAPI"),
            ChannelSpec(name="GFP", index=1, stain="GFP"),
        ))
        self.assertEqual(img.channel("gfp").index, 1)
        self.assertEqual(img.channel("DAPI").index, 0)
        self.assertEqual(img.channel_of_stain("gfp").name, "GFP")
        self.assertEqual(img.channel_names, ("C0", "GFP"))
        self.assertEqual(img.plane_count, 2)

    def test_unknown_channel_raises_instead_of_defaulting_to_zero(self):
        """Silently falling back to channel 0 is how a nuclear stain ends up
        being measured as if it were a membrane marker."""
        img = self._image(channels=(ChannelSpec(name="C0", index=0),))
        with self.assertRaises(KeyError):
            img.channel("GFP")

    def test_non_contiguous_channel_indices_rejected(self):
        with self.assertRaises(ValueError):
            self._image(channels=(ChannelSpec(name="A", index=0), ChannelSpec(name="B", index=2)))

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            self._image(pixel_size_um=0)
        with self.assertRaises(ValueError):
            self._image(n_z=0)
        with self.assertRaises(ValueError):
            self._image(bit_depth=0)

    def test_serialisation_shape(self):
        payload = self._image(pixel_size_um=0.5).to_dict()
        self.assertEqual(payload["width_px"], 2048)
        self.assertEqual(payload["modality"], "unknown")


class TestMeasurement(unittest.TestCase):
    def test_uncalibrated_measurement_is_not_physically_calibrated(self):
        m = Measurement(area_px=1234, circularity=0.8)
        self.assertFalse(m.is_physically_calibrated)
        self.assertIsNone(m.area_um2)

    def test_calibrated_measurement_flags_true(self):
        m = Measurement(area_px=1234, area_um2=42.0)
        self.assertTrue(m.is_physically_calibrated)

    def test_single_object_predicate(self):
        self.assertTrue(Measurement(area_px=10).is_single_object)
        self.assertFalse(Measurement(area_px=10, n_components=3, is_multi_component=True).is_single_object)

    def test_serialisation_keeps_channel_dicts(self):
        m = Measurement(area_px=5, mean_intensity={"GFP": 1200.0}, max_intensity={"GFP": 5000.0})
        payload = m.to_dict()
        self.assertEqual(payload["mean_intensity"], {"GFP": 1200.0})
        self.assertEqual(payload["max_intensity"], {"GFP": 5000.0})


class TestInstance(unittest.TestCase):
    def _instance(self, **kw):
        defaults = dict(
            id=new_id("inst"), image_id="img1", class_key="organoid0",
            taxonomy_name="organoid_4class",
        )
        defaults.update(kw)
        return Instance(**defaults)

    def test_defaults_and_derived(self):
        inst = self._instance()
        self.assertFalse(inst.has_mask)
        self.assertFalse(inst.is_tracked)
        self.assertEqual(inst.provenance, Provenance.HUMAN)
        self.assertEqual(inst.review_status, ReviewStatus.UNREVIEWED)

    def test_confidence_bounds_enforced(self):
        with self.assertRaises(ValueError):
            self._instance(confidence=1.5)
        with self.assertRaises(ValueError):
            self._instance(confidence=-0.1)
        self.assertEqual(self._instance(confidence=0.5).confidence, 0.5)

    def test_negative_time_or_z_rejected(self):
        with self.assertRaises(ValueError):
            self._instance(z_index=-1)
        with self.assertRaises(ValueError):
            self._instance(t_index=-1)

    def test_required_fields(self):
        with self.assertRaises(ValueError):
            Instance(id="", image_id="i", class_key="c", taxonomy_name="t")
        with self.assertRaises(ValueError):
            Instance(id="i", image_id="", class_key="c", taxonomy_name="t")
        with self.assertRaises(ValueError):
            Instance(id="i", image_id="i", class_key="", taxonomy_name="t")

    def test_enums_accept_plain_strings(self):
        inst = self._instance(provenance="consensus", review_status="adjudicated")
        self.assertIs(inst.provenance, Provenance.CONSENSUS)
        self.assertTrue(inst.is_ground_truth)

    def test_mask_is_excluded_from_serialisation(self):
        import numpy as np

        inst = self._instance(mask=np.ones((4, 4), dtype=bool))
        payload = inst.to_dict()
        self.assertNotIn("mask", payload)

    def test_with_mask_returns_a_new_frozen_instance(self):
        import numpy as np

        inst = self._instance()
        masked = inst.with_mask(np.ones((2, 2), dtype=bool))
        self.assertFalse(inst.has_mask)
        self.assertTrue(masked.has_mask)


class TestSpecimenAndTrack(unittest.TestCase):
    def test_specimen_rejects_negative_passage(self):
        with self.assertRaises(ValueError):
            Specimen(id="s1", passage=-1)
        self.assertEqual(Specimen(id="s1", passage=3).to_dict()["passage"], 3)

    def test_track_accumulates_and_reports_span(self):
        track = Track(id="t1")
        track.add("i1", 0)
        self.assertEqual((track.start_t, track.end_t, track.length), (0, 0, 1))
        track.add("i2", 5)
        self.assertEqual((track.start_t, track.end_t, track.length), (0, 5, 2))
        self.assertEqual(track.duration_frames, 5)
        self.assertTrue(track.is_root)

    def test_track_records_division_lineage(self):
        parent = Track(id="t_parent")
        parent.add("i1", 0)
        parent.record_division(3, "t_child")
        self.assertTrue(parent.is_division)
        self.assertEqual(parent.division_events, [3])
        self.assertEqual(parent.children, ["t_child"])
        # recording the same child twice must not duplicate it
        parent.record_division(3, "t_child")
        self.assertEqual(parent.children, ["t_child"])

        child = Track(id="t_child", lineage_parent="t_parent")
        child.add("i2", 3)
        self.assertFalse(child.is_root)

    def test_track_serialisation(self):
        track = Track(id="t1")
        track.add("i1", 0)
        payload = track.to_dict()
        self.assertEqual(payload["length"], 1)
        self.assertEqual(payload["instance_ids"], ["i1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
