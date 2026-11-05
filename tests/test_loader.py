# ── tests/test_loader.py ──
"""Tests for the bit-depth-preserving loader.

The behaviours asserted here are the ones that were broken by the old
``data/multiorg_tiling.py:86-107`` path: per-image min-max normalisation,
channel collapse, and irreversible 8-bit quantisation at load time.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (
    DatasetNormalizer,
    LoadError,
    NormalizationStrategy,
    infer_bit_depth,
    load_image,
    read_raw,
    sha256_file,
    to_display_rgb,
    to_display_uint8,
)


def _write_tiff(path, array):
    """Write a TIFF, pinning ``photometric`` so stacks stay plain pages.

    Without this, tifffile stores a ``(pages, y, x)`` uint16 array as planar RGB
    and warns that the default will change in a future release — which would
    make these fixtures silently mean something different.
    """
    try:
        import tifffile

        photometric = "minisblack" if array.ndim >= 3 else None
        tifffile.imwrite(path, array, photometric=photometric)
        return True
    except ImportError:
        from PIL import Image as PILImage

        PILImage.fromarray(array).save(path)
        return True


class TempImages(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestBitDepth(unittest.TestCase):
    def test_infers_from_dtype_not_from_maximum(self):
        """A 16-bit image with a dim maximum is still a 16-bit image."""
        arr = np.full((4, 4), 900, dtype=np.uint16)
        self.assertEqual(infer_bit_depth(arr), 16)
        self.assertEqual(infer_bit_depth(np.zeros((2, 2), dtype=np.uint8)), 8)
        self.assertEqual(infer_bit_depth(np.zeros((2, 2), dtype=np.uint32)), 32)

    def test_float_input_is_rejected(self):
        with self.assertRaises(LoadError):
            infer_bit_depth(np.zeros((2, 2), dtype=np.float32))


class TestLoadImage(TempImages):
    def test_16_bit_values_survive_loading_unchanged(self):
        source = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 7)
        path = self.path("a16.tiff")
        _write_tiff(path, source)

        loaded = load_image(path, pixel_size_um=0.325)

        # The whole point: no rescaling happened on the way in.
        np.testing.assert_array_equal(loaded.array[0, 0, 0], source)
        self.assertEqual(loaded.bit_depth, 16)
        self.assertEqual(loaded.dtype, np.uint16)
        self.assertTrue(loaded.is_quantitative)
        self.assertEqual(loaded.array.shape, (1, 1, 1, 64, 64))

    def test_layout_is_always_z_t_c_y_x(self):
        loaded = load_image(self._single_plane_path())
        self.assertEqual(loaded.array.ndim, 5)
        self.assertEqual((loaded.n_z, loaded.n_t, loaded.n_channels), (1, 1, 1))

    def test_multichannel_zstack_is_decomposed_correctly(self):
        pages = np.stack([np.full((16, 16), 100 * i, np.uint16) for i in range(6)])
        path = self.path("stack.tiff")
        _write_tiff(path, pages)

        loaded = load_image(
            path, n_z=3, n_channels=2,
            channel_names=["DAPI", "GFP"], stains=["DAPI", "GFP"],
            pixel_size_um=0.5, z_step_um=1.0,
        )
        self.assertEqual(loaded.array.shape, (3, 1, 2, 16, 16))
        self.assertTrue(loaded.image.is_3d)
        self.assertFalse(loaded.image.is_4d)
        # channels must be distinct planes, not one plane replicated
        self.assertNotEqual(int(loaded.array[0, 0, 0].max()), int(loaded.array[0, 0, 1].max()))

    def test_ambiguous_page_count_is_refused_not_guessed(self):
        pages = np.stack([np.full((8, 8), i, np.uint16) for i in range(6)])
        path = self.path("ambig.tiff")
        _write_tiff(path, pages)
        with self.assertRaises(LoadError) as ctx:
            load_image(path, n_channels=1)
        self.assertIn("ambiguous", str(ctx.exception).lower())

    def test_page_layout_declares_the_ambiguous_case(self):
        pages = np.stack([np.full((8, 8), i, np.uint16) for i in range(6)])
        path = self.path("ambig2.tiff")
        _write_tiff(path, pages)

        as_z = load_image(path, page_layout="z", n_channels=1)
        self.assertEqual(as_z.n_z, 6)

        as_t = load_image(path, page_layout="t", n_channels=1)
        self.assertEqual(as_t.n_t, 6)

    def test_inconsistent_declared_dimensions_raise(self):
        path = self._single_plane_path()
        with self.assertRaises(LoadError):
            load_image(path, n_z=3, n_t=2, n_channels=2)

    def test_channel_metadata_mismatch_raises(self):
        pages = np.stack([np.full((8, 8), i, np.uint16) for i in range(4)])
        path = self.path("c4.tiff")
        _write_tiff(path, pages)
        with self.assertRaises(LoadError):
            load_image(path, n_z=1, n_channels=4, channel_names=["only_one"])

    def test_missing_file_raises(self):
        with self.assertRaises(LoadError):
            load_image(self.path("nope.tiff"))

    def test_uncalibrated_load_leaves_physical_fields_empty(self):
        loaded = load_image(self._single_plane_path())
        self.assertFalse(loaded.image.is_calibrated)
        self.assertIsNone(loaded.image.extent_um)

    def test_checksum_is_computed_on_request(self):
        path = self._single_plane_path()
        loaded = load_image(path, checksum=True)
        self.assertEqual(loaded.image.checksum, sha256_file(path))

    def test_quantify_is_a_separate_step_from_loading(self):
        loaded = load_image(self._single_plane_path())
        normaliser = DatasetNormalizer(
            strategy=NormalizationStrategy.NATIVE_SCALE, bit_depth=16, n_channels=1
        )
        quantified = loaded.quantify(normaliser)
        self.assertEqual(quantified.shape, loaded.array.shape)
        self.assertTrue(np.issubdtype(quantified.dtype, np.floating))
        self.assertLessEqual(float(quantified.max()), 1.0)

    def _single_plane_path(self):
        path = self.path("plane.tiff")
        _write_tiff(path, (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 3))
        return path


class TestDatasetNormalizer(unittest.TestCase):
    def setUp(self):
        self.reference = [
            np.stack([np.full((8, 8), 100, np.uint16), np.full((8, 8), 200, np.uint16)])
            for _ in range(4)
        ]

    def test_fitted_on_reference_keeps_images_comparable(self):
        """The core fix: statistics come from a reference set, not each image."""
        normaliser = DatasetNormalizer.fit(
            self.reference, strategy=NormalizationStrategy.DATASET_MINMAX
        )
        dim = np.stack([np.full((8, 8), 100, np.uint16), np.full((8, 8), 120, np.uint16)])
        bright = np.stack([np.full((8, 8), 900, np.uint16), np.full((8, 8), 950, np.uint16)])

        dim_out = normaliser.apply(dim, channel_axis=0)
        bright_out = normaliser.apply(bright, channel_axis=0)

        # A dim image stays dim; a saturated one clips. Per-image min-max would
        # have mapped both onto [0, 1] and destroyed the difference.
        self.assertLess(float(dim_out.max()), 0.3)
        self.assertGreater(float(bright_out.min()), 0.9)

    def test_per_image_fit_collapses_the_difference(self):
        """Documenting the behaviour we replaced, so its absence is a real test."""
        dim = np.stack([np.full((8, 8), 100, np.uint16), np.full((8, 8), 120, np.uint16)])
        bright = np.stack([np.full((8, 8), 900, np.uint16), np.full((8, 8), 950, np.uint16)])

        dim_norm = DatasetNormalizer.fit_per_image(
            dim, strategy=NormalizationStrategy.DATASET_MINMAX
        )
        bright_norm = DatasetNormalizer.fit_per_image(
            bright, strategy=NormalizationStrategy.DATASET_MINMAX
        )
        dim_out = dim_norm.apply(dim, channel_axis=0)
        bright_out = bright_norm.apply(bright, channel_axis=0)

        self.assertAlmostEqual(float(dim_out.max()), float(bright_out.max()), places=5)

    def test_percentile_strategy_ignores_outliers_when_fitting(self):
        """A single hot pixel must not define the intensity scale."""
        stack = np.full((8, 8), 500, np.uint16)
        stack[0, 0] = 60000  # a hot pixel
        normaliser = DatasetNormalizer.fit(
            [stack], strategy=NormalizationStrategy.DATASET_PERCENTILE,
            low_percentile=0.0, high_percentile=90.0,
        )
        # The fitted upper bound stayed near the bulk of the data.
        self.assertLess(float(normaliser.high[0]), 30000.0)

        out = normaliser.apply(stack, channel_axis=0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        # The outlier saturates rather than stretching the scale for everyone.
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)

    def test_minmax_strategy_does_let_an_outlier_stretch_the_scale(self):
        """Contrast case: this is why percentile is the default."""
        stack = np.full((8, 8), 500, np.uint16)
        stack[0, 0] = 60000
        normaliser = DatasetNormalizer.fit(
            [stack], strategy=NormalizationStrategy.DATASET_MINMAX
        )
        self.assertAlmostEqual(float(normaliser.high[0]), 60000.0, places=1)

    def test_channel_count_mismatch_raises(self):
        normaliser = DatasetNormalizer.fit(
            self.reference, strategy=NormalizationStrategy.DATASET_MINMAX
        )
        with self.assertRaises(LoadError):
            normaliser.apply(np.zeros((3, 4, 4), np.uint16), channel_axis=0)

    def test_fit_rejects_inconsistent_reference_channels(self):
        with self.assertRaises(LoadError):
            DatasetNormalizer.fit(
                [np.zeros((2, 4, 4), np.uint16), np.zeros((3, 4, 4), np.uint16)],
                strategy=NormalizationStrategy.DATASET_MINMAX,
            )

    def test_fit_needs_at_least_one_reference(self):
        with self.assertRaises(LoadError):
            DatasetNormalizer.fit([], strategy=NormalizationStrategy.DATASET_MINMAX)

    def test_native_scale_needs_no_fitted_statistics(self):
        normaliser = DatasetNormalizer(
            strategy=NormalizationStrategy.NATIVE_SCALE, bit_depth=16, n_channels=1
        )
        out = normaliser.apply(np.full((8, 8), 65535, np.uint16)[None], channel_axis=0)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_serialisation_is_round_trippable_for_audit(self):
        normaliser = DatasetNormalizer.fit(
            self.reference, strategy=NormalizationStrategy.DATASET_MINMAX
        )
        payload = normaliser.to_dict()
        self.assertEqual(payload["strategy"], "dataset_minmax")
        self.assertEqual(len(payload["low"]), 2)


class TestDisplayConversion(unittest.TestCase):
    def test_single_channel_is_not_expanded_into_invented_channels(self):
        plane = (np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) * 5)
        rgb = to_display_rgb(plane)
        self.assertEqual(rgb.shape, (32, 32, 3))
        # grayscale triple must be internally consistent
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_multichannel_picks_real_channels(self):
        """Two channels with different spatial structure must render differently.

        Note an affine pair (``a`` and ``a * k``) would normalise to the same
        image, which is correct behaviour — percentile stretching is affine. So
        the fixture uses structurally different planes.
        """
        ramp = np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) * 5
        block = np.zeros((8, 8), np.uint16)
        block[2:6, 2:6] = 900
        rgb = to_display_rgb(np.stack([ramp, block]), channel_indices=[0, 1])
        self.assertFalse(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]))

    def test_too_many_channel_indices_rejected(self):
        arr = np.zeros((4, 8, 8), np.uint16)
        with self.assertRaises(LoadError):
            to_display_rgb(arr, channel_indices=[0, 1, 2, 3])

    def test_uint8_conversion_is_bounded(self):
        out = to_display_uint8(np.arange(100, dtype=np.uint16) * 600)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLessEqual(int(out.max()), 255)

    def test_flat_image_does_not_divide_by_zero(self):
        out = to_display_uint8(np.full((4, 4), 7, np.uint16))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.isfinite(out).all())


class TestReadRaw(TempImages):
    def test_raw_read_preserves_dtype_and_does_not_rescale(self):
        source = (np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 11)
        path = self.path("raw.tiff")
        _write_tiff(path, source)
        raw = read_raw(path)
        self.assertEqual(raw.dtype, np.uint16)
        self.assertEqual(int(raw.max()), int(source.max()))
        self.assertEqual(int(raw.min()), int(source.min()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
