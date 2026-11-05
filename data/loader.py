# ── data/loader.py ──
"""Bit-depth-preserving, channel-aware image loading.

The defect this replaces
-----------------------
``data/multiorg_tiling.py:86-107`` was the only 16-bit entry point in the
repository, and it did three destructive things on the way in:

1. **Per-image min-max normalisation.** ``(arr - arr.min()) / (arr.max() - arr.min())``
   is computed independently for every field of view. Two plates imaged on the
   same microscope with different exposure therefore land on the same 0-255
   range regardless of their actual signal, so intensity is not comparable
   *across* images or sites. In a federated setting this manufactures domain
   shift that was not present in the specimen.
2. **Channel collapse.** ``np.stack([arr8, arr8, arr8])`` replicates one plane
   into three channels, so a single-channel grayscale plane becomes a fake RGB
   image and any genuinely multi-channel acquisition loses its second marker.
3. **Irreversible 8-bit quantisation.** 12- and 16-bit acquisitions were pushed
   to 256 levels before anything was measured, discarding the dynamic range
   that fluorescence quantification depends on.

This module keeps the quantitative array at its native bit depth and channel
count, and makes display conversion a separate, explicitly-invoked step.

Canonical layout
----------------
Arrays are returned as ``(Z, T, C, Y, X)`` — always 5-D, always
``float`` for quantitative work. Singleton axes are kept rather than squeezed
so indexing code cannot silently depend on which dimensions happen to be >1.

Ambiguity policy
----------------
A multi-page TIFF does not say whether its pages are Z-planes or time points.
Guessing produces silently mislabelled data, so this module requires either
explicit ``page_layout`` or a page count that admits exactly one reading.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from analysis.schema import ChannelSpec, Image, Modality

__all__ = [
    "DatasetNormalizer",
    "LoadError",
    "LoadedImage",
    "NormalizationStrategy",
    "infer_bit_depth",
    "load_image",
    "read_raw",
    "sha256_file",
    "to_display_rgb",
    "to_display_uint8",
]


class LoadError(RuntimeError):
    """Raised when an image cannot be read unambiguously."""


# ─────────────────────────────────────────────────────────────────────────────
# Bit depth
# ─────────────────────────────────────────────────────────────────────────────


def infer_bit_depth(array: np.ndarray) -> int:
    """Infer the acquisition bit depth of an integer array.

    Uses the dtype, not the observed maximum: a 16-bit image whose brightest
    pixel happens to be 900 is still a 16-bit image, and rescaling it as if it
    were 10-bit would corrupt the calibration.
    """
    if np.issubdtype(array.dtype, np.floating):
        raise LoadError(
            f"infer_bit_depth expects an integer array, got {array.dtype}. "
            "Convert display data back to an integer type before storing it."
        )
    return int(array.dtype.itemsize * 8)


def _native_max(bit_depth: int) -> float:
    return float((1 << bit_depth) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────


class NormalizationStrategy(str, Enum):
    """How raw intensities are mapped for quantitative use.

    ``NONE`` is the default and the only lossless option: values stay on their
    native scale. The others are explicit opt-ins because each one trades
    interpretability for comparability, and the trade must be visible at the
    call site.
    """

    NONE = "none"
    #: Divide by the dtype's native max. Preserves relative intensity across
    #: everything acquired at the same bit depth.
    NATIVE_SCALE = "native_scale"
    #: Map an explicit ``[low, high]`` intensity window to [0, 1].
    FIXED_RANGE = "fixed_range"
    #: Per-channel z-score using statistics fitted on a reference set.
    DATASET_ZSCORE = "dataset_zscore"
    #: Per-channel min-max using statistics fitted on a reference set.
    DATASET_MINMAX = "dataset_minmax"
    #: Per-channel percentile clip using statistics fitted on a reference set.
    DATASET_PERCENTILE = "dataset_percentile"


@dataclass
class DatasetNormalizer:
    """Fitted, reusable intensity normalisation.

    The whole point is that the mapping is computed **once on a reference set**
    and then applied unchanged to every image — including images from other
    federated sites. That is what makes intensities comparable across nodes,
    which per-image min-max can never achieve.

    Usage::

        norm = DatasetNormalizer.fit(reference_arrays, strategy=...)
        arr = norm.apply(raw)

    Fitting on a single array is supported but emits a warning-worthy signal in
    review: a per-image fit is exactly the behaviour this module exists to
    replace, so :meth:`fit_per_image` names that case explicitly.
    """

    strategy: NormalizationStrategy = NormalizationStrategy.NATIVE_SCALE
    bit_depth: int = 8
    #: Per-channel statistics, shaped ``(C,)`` or ``(C, ...)``.
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    n_channels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, NormalizationStrategy):
            self.strategy = NormalizationStrategy(self.strategy)

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        references: Iterable[np.ndarray],
        *,
        strategy: NormalizationStrategy = NormalizationStrategy.DATASET_PERCENTILE,
        low_percentile: float = 0.1,
        high_percentile: float = 99.9,
        channel_axis: int = 0,
        bit_depth: Optional[int] = None,
    ) -> "DatasetNormalizer":
        """Fit statistics across a set of images.

        Args:
            references: arrays sharing a channel count; each is reduced along
                every axis except ``channel_axis``.
            strategy: which statistic to retain.
            low_percentile / high_percentile: used by ``DATASET_PERCENTILE``.
            channel_axis: axis index in each reference that is the channel axis.
            bit_depth: override when the references are not integers.
        """
        arrays = [np.asarray(a) for a in references]
        if not arrays:
            raise LoadError("DatasetNormalizer.fit needs at least one reference array")

        n_channels = arrays[0].shape[channel_axis]
        for i, a in enumerate(arrays):
            if a.shape[channel_axis] != n_channels:
                raise LoadError(
                    f"Reference {i} has {a.shape[channel_axis]} channels but the "
                    f"first has {n_channels}; cannot fit a shared normaliser"
                )

        if not isinstance(strategy, NormalizationStrategy):
            strategy = NormalizationStrategy(strategy)

        if bit_depth is None:
            bit_depth = infer_bit_depth(arrays[0]) if np.issubdtype(arrays[0].dtype, np.integer) else 16

        normaliser = cls(strategy=strategy, bit_depth=bit_depth, n_channels=n_channels)

        # Move the channel axis last so the reduction is uniform.
        moved = [np.moveaxis(a, channel_axis, -1).reshape(-1, n_channels) for a in arrays]
        pooled = np.concatenate(moved, axis=0).astype(np.float64)

        if strategy is NormalizationStrategy.DATASET_PERCENTILE:
            normaliser.low = np.percentile(pooled, low_percentile, axis=0)
            normaliser.high = np.percentile(pooled, high_percentile, axis=0)
        elif strategy is NormalizationStrategy.DATASET_MINMAX:
            normaliser.low = pooled.min(axis=0).astype(np.float64)
            normaliser.high = pooled.max(axis=0).astype(np.float64)
        elif strategy is NormalizationStrategy.DATASET_ZSCORE:
            normaliser.mean = pooled.mean(axis=0)
            normaliser.std = pooled.std(axis=0)
        elif strategy is NormalizationStrategy.FIXED_RANGE:
            raise LoadError(
                "FIXED_RANGE needs explicit bounds; construct "
                "DatasetNormalizer(strategy=FIXED_RANGE, low=..., high=...) directly"
            )
        # NONE and NATIVE_SCALE need no fitted statistics.

        return normaliser

    @classmethod
    def fit_per_image(
        cls,
        array: np.ndarray,
        *,
        strategy: NormalizationStrategy = NormalizationStrategy.DATASET_PERCENTILE,
        channel_axis: int = 0,
        **kwargs: Any,
    ) -> "DatasetNormalizer":
        """Fit on one image. Named so the per-image behaviour is auditable.

        Using this on the training path reintroduces the cross-image
        incomparability this module exists to remove — prefer :meth:`fit` on a
        reference set.
        """
        return cls.fit([array], strategy=strategy, channel_axis=channel_axis, **kwargs)

    # ── application ──────────────────────────────────────────────────────

    def apply(self, array: np.ndarray, *, channel_axis: int = 0) -> np.ndarray:
        """Apply the fitted mapping. Returns ``float32``, same shape as input."""
        arr = np.asarray(array)
        if arr.shape[channel_axis] != self.n_channels:
            raise LoadError(
                f"Normaliser fitted for {self.n_channels} channels but the array "
                f"has {arr.shape[channel_axis]}"
            )
        out = arr.astype(np.float32)

        if self.strategy is NormalizationStrategy.NONE:
            return out
        if self.strategy is NormalizationStrategy.NATIVE_SCALE:
            return out / _native_max(self.bit_depth)
        if self.strategy is NormalizationStrategy.DATASET_ZSCORE:
            moved = np.moveaxis(out, channel_axis, -1)
            moved = (moved - self.mean.astype(np.float32)) / np.where(
                self.std.astype(np.float32) == 0, 1.0, self.std.astype(np.float32)
            )
            return np.moveaxis(moved, -1, channel_axis)

        if self.low is None or self.high is None:
            raise LoadError(f"{self.strategy.value} requires fitted low/high bounds")

        moved = np.moveaxis(out, channel_axis, -1)
        span = (self.high - self.low).astype(np.float32)
        span = np.where(span == 0, 1.0, span)
        moved = (moved - self.low.astype(np.float32)) / span
        moved = np.clip(moved, 0.0, 1.0)
        return np.moveaxis(moved, -1, channel_axis)

    def to_dict(self) -> dict:
        def _as_list(v: Optional[np.ndarray]) -> Optional[list]:
            return None if v is None else np.asarray(v).ravel().tolist()

        return {
            "strategy": self.strategy.value,
            "bit_depth": self.bit_depth,
            "n_channels": self.n_channels,
            "low": _as_list(self.low),
            "high": _as_list(self.high),
            "mean": _as_list(self.mean),
            "std": _as_list(self.std),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Raw reading
# ─────────────────────────────────────────────────────────────────────────────


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, for provenance records."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_with_tifffile(path: str) -> np.ndarray:
    import tifffile  # type: ignore

    return np.asarray(tifffile.imread(path))


def _read_with_pil(path: str) -> np.ndarray:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        n_frames = getattr(im, "n_frames", 1)
        if n_frames == 1:
            return np.asarray(im)
        frames = []
        for i in range(n_frames):
            im.seek(i)
            frames.append(np.asarray(im))
        return np.stack(frames, axis=0)


def read_raw(path: str, *, prefer_tifffile: bool = True) -> np.ndarray:
    """Read an image file without any rescaling, keeping its native dtype.

    Multi-page files come back stacked on a leading axis. Values are handed
    back exactly as stored — no normalisation, no dtype conversion.
    """
    if not os.path.isfile(path):
        raise LoadError(f"Image not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    is_tiff = ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff")

    if is_tiff and prefer_tifffile:
        try:
            return _read_with_tifffile(path)
        except ImportError:
            pass

    try:
        return _read_with_pil(path)
    except Exception as exc:  # pragma: no cover - depends on the codec
        raise LoadError(f"Could not read {path}: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Page layout resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_layout(
    n_pages: int,
    *,
    n_channels: int,
    page_layout: Optional[str],
) -> tuple[int, int, int]:
    """Determine ``(n_z, n_t, n_c)`` for a stacked multi-page array.

    ``page_layout`` is a permutation string describing the page ordering from
    slowest to fastest varying. A single axis is accepted as shorthand — ``"z"``
    and ``"t"`` are the two genuinely ambiguous cases, so they carry most of the
    value. Any axis left unspecified is taken to be 1.
    """
    if page_layout is not None:
        layout = page_layout.strip().lower()
        if not layout or any(axis not in "ztc" for axis in layout):
            raise LoadError(
                f"page_layout must name axes from 'ztc', got {page_layout!r}"
            )
        if len(set(layout)) != len(layout):
            raise LoadError(f"page_layout must not repeat an axis, got {page_layout!r}")

        # Walk the layout from the fastest-varying axis inwards, consuming the
        # page count. The channel axis is fixed by the data; whichever of Z/T
        # is named takes the remainder, and an omitted axis stays at 1.
        remaining = n_pages
        resolved: dict[str, int] = {"z": 1, "t": 1, "c": n_channels}
        for axis in reversed(layout):
            if axis == "c":
                if remaining % n_channels:
                    raise LoadError(
                        f"{n_pages} pages is not divisible by {n_channels} channels"
                    )
                resolved["c"] = n_channels
                remaining //= n_channels
            else:
                resolved[axis] = max(1, remaining)
                remaining = 1
        if remaining != 1:
            raise LoadError(f"page_layout {page_layout!r} does not divide {n_pages} pages evenly")
        return resolved["z"], resolved["t"], resolved["c"]

    # No hint: only accept unambiguous readings.
    if n_pages == n_channels:
        return 1, 1, n_channels
    if n_pages == 1:
        return 1, 1, 1
    if n_channels == 1:
        raise LoadError(
            f"{n_pages} pages with 1 channel is ambiguous: could be a {n_pages}-plane "
            "Z-stack or a time series. Pass page_layout='z' or page_layout='t' to say which."
        )
    raise LoadError(
        f"{n_pages} pages across {n_channels} channels is ambiguous. "
        "Pass page_layout (e.g. 'ztc', 'tzc') to declare the acquisition order."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LoadedImage
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LoadedImage:
    """A loaded acquisition plus its metadata.

    Attributes:
        array: quantitative data, shape ``(Z, T, C, Y, X)``, native dtype.
        image: acquisition metadata (:class:`analysis.schema.Image`).
        normalizer: the normaliser applied, if any, kept so a reviewer can
            reconstruct exactly how intensities were mapped.
    """

    array: np.ndarray
    image: Image
    normalizer: Optional[DatasetNormalizer] = None

    @property
    def n_z(self) -> int:
        return self.array.shape[0]

    @property
    def n_t(self) -> int:
        return self.array.shape[1]

    @property
    def n_channels(self) -> int:
        return self.array.shape[2]

    @property
    def height(self) -> int:
        return self.array.shape[3]

    @property
    def width(self) -> int:
        return self.array.shape[4]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def bit_depth(self) -> int:
        return self.image.bit_depth

    @property
    def is_quantitative(self) -> bool:
        """True when the array still holds native-scale integer intensities."""
        return np.issubdtype(self.array.dtype, np.integer)

    def plane(self, z: int = 0, t: int = 0, c: int = 0) -> np.ndarray:
        """Return one 2-D plane as ``(Y, X)``."""
        return self.array[z, t, c]

    def channel(self, index: int) -> np.ndarray:
        """Return every plane of one channel as ``(Z, T, Y, X)``."""
        return self.array[:, :, index]

    def quantify(self, normalizer: DatasetNormalizer) -> np.ndarray:
        """Return a normalised copy for quantitative comparison.

        Shape is unchanged; the returned array is ``float32``.
        """
        moved = np.moveaxis(self.array, 2, 0)  # (C, Z, T, Y, X)
        out = normalizer.apply(moved, channel_axis=0)
        return np.moveaxis(out, 0, 2)

    def to_dict(self) -> dict:
        return {
            "array_shape": list(self.array.shape),
            "array_dtype": str(self.array.dtype),
            "image": self.image.to_dict(),
            "normalizer": self.normalizer.to_dict() if self.normalizer else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_image(
    path: str,
    *,
    image_id: Optional[str] = None,
    specimen_id: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    stains: Optional[Sequence[Optional[str]]] = None,
    wavelengths_nm: Optional[Sequence[Optional[float]]] = None,
    modality: Modality | str = Modality.UNKNOWN,
    page_layout: Optional[str] = None,
    pixel_size_um: Optional[float] = None,
    z_step_um: Optional[float] = None,
    time_interval_min: Optional[float] = None,
    n_z: Optional[int] = None,
    n_t: Optional[int] = None,
    n_channels: Optional[int] = None,
    plate: Optional[str] = None,
    well: Optional[str] = None,
    objective: Optional[str] = None,
    checksum: bool = False,
) -> LoadedImage:
    """Load one acquisition into canonical ``(Z, T, C, Y, X)`` form.

    Nothing is rescaled and nothing is dropped: the returned array carries the
    file's native dtype, bit depth and channel count.

    Args:
        path: image file.
        channel_names: channel identifiers, e.g. ``("DAPI", "GFP")``. Defaults
            to ``C0``, ``C1``, ... which is honest about the fact that the file
            alone does not name its channels.
        stains: what each channel labels.
        page_layout: permutation of ``"ztc"`` declaring the page order of a
            multi-page file. Required when the page count is ambiguous.
        pixel_size_um: physical pixel size. Leave ``None`` rather than guessing.
        checksum: compute a SHA-256 of the file for the provenance record.

    Raises:
        LoadError: on an unreadable file or an ambiguous page layout.
    """
    raw = read_raw(path)

    bit_depth = infer_bit_depth(raw) if np.issubdtype(raw.dtype, np.integer) else 16

    # ── normalise the raw array into (pages, y, x) or (y, x, c) ──
    declared_channels = n_channels
    if raw.ndim == 2:
        stacked = raw[np.newaxis, ...]
        layout_c = 1
    elif raw.ndim == 3:
        # Ambiguous for TIFF stacks: (pages, y, x) or (y, x, channels)?
        # A trailing axis of 3 or 4 on an 8-bit file is conventionally RGB(A).
        if raw.shape[-1] in (3, 4) and raw.dtype == np.uint8:
            stacked = np.moveaxis(raw, -1, 0)
            layout_c = raw.shape[-1]
        else:
            stacked = raw
            layout_c = 1
    elif raw.ndim == 4:
        # (pages, y, x, c)
        stacked = np.moveaxis(raw, -1, 0)
        layout_c = raw.shape[-1]
    else:
        raise LoadError(f"Unsupported array rank {raw.ndim} for {path}")

    if declared_channels is not None:
        layout_c = declared_channels

    n_pages = stacked.shape[0]

    if n_z is not None or n_t is not None:
        nz = int(n_z or 1)
        nt = int(n_t or 1)
        nc = int(declared_channels or layout_c)
        if nz * nt * nc != n_pages:
            raise LoadError(
                f"Declared (n_z={nz}, n_t={nt}, n_c={nc}) implies {nz * nt * nc} planes "
                f"but the file holds {n_pages}"
            )
    else:
        nz, nt, nc = _resolve_layout(n_pages, n_channels=layout_c, page_layout=page_layout)

    if nc * nt * nz != n_pages:
        raise LoadError(
            f"Resolved layout (n_z={nz}, n_t={nt}, n_c={nc}) needs {nz * nt * nc} "
            f"planes but the file holds {n_pages}"
        )

    # Pages are ordered z fastest, then t, then c.
    array = stacked.reshape(nz, nt, nc, stacked.shape[1], stacked.shape[2])

    # ── channel metadata ──
    names = list(channel_names) if channel_names else [f"C{i}" for i in range(nc)]
    if len(names) != nc:
        raise LoadError(f"Got {len(names)} channel_names for {nc} channels")
    stain_seq = list(stains) if stains else [None] * nc
    wave_seq = list(wavelengths_nm) if wavelengths_nm else [None] * nc
    if len(stain_seq) != nc or len(wave_seq) != nc:
        raise LoadError("stains / wavelengths_nm must match the channel count")

    channels = tuple(
        ChannelSpec(
            name=names[i],
            index=i,
            stain=stain_seq[i],
            wavelength_nm=wave_seq[i],
            is_reference=(i == 0),
        )
        for i in range(nc)
    )

    image = Image(
        id=image_id or _default_image_id(path),
        path=os.path.abspath(path),
        specimen_id=specimen_id,
        width_px=int(array.shape[4]),
        height_px=int(array.shape[3]),
        channels=channels,
        modality=modality if isinstance(modality, Modality) else Modality(modality),
        bit_depth=bit_depth,
        n_z=nz,
        n_t=nt,
        pixel_size_um=pixel_size_um,
        z_step_um=z_step_um,
        time_interval_min=time_interval_min,
        plate=plate,
        well=well,
        objective=objective,
        checksum=sha256_file(path) if checksum else None,
    )

    return LoadedImage(array=array, image=image)


def _default_image_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"img_{stem}"


# ─────────────────────────────────────────────────────────────────────────────
# Display conversion — deliberately separate from the quantitative path
# ─────────────────────────────────────────────────────────────────────────────


def to_display_uint8(
    array: np.ndarray,
    *,
    normalizer: Optional[DatasetNormalizer] = None,
    percentiles: tuple[float, float] = (0.1, 99.9),
) -> np.ndarray:
    """Convert an array to ``uint8`` **for display only**.

    Never measure on the result: the mapping is windowed and clipped, so
    intensities are no longer proportional. Use :func:`load_image`'s array or
    :meth:`LoadedImage.quantify` for anything quantitative.

    When ``normalizer`` is supplied its fitted bounds are used, so tiles from
    different images stay visually comparable. Without it the bounds come from
    this array alone, which is the old per-image behaviour and is called out
    here so the difference is a deliberate choice.
    """
    arr = np.asarray(array).astype(np.float64)

    if normalizer is not None and normalizer.low is not None and normalizer.high is not None:
        low, high = normalizer.low, normalizer.high
    else:
        low = np.percentile(arr, percentiles[0])
        high = np.percentile(arr, percentiles[1])

    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    span = np.where(high - low == 0, 1.0, high - low)
    scaled = (arr - low) / span
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def to_display_rgb(
    array: np.ndarray,
    *,
    normalize: bool = True,
    channel_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Render a plane or channel set as an RGB ``uint8`` image for display.

    Unlike the old ``convert_tiff_to_rgb``, a single-channel input is rendered
    as a genuine grayscale RGB triple of *that* channel — the function never
    invents channels that were not acquired, and multi-channel input picks real
    channels rather than duplicating one.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        gray = to_display_uint8(arr) if normalize else arr.astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    if arr.ndim != 3:
        raise LoadError(f"to_display_rgb expects (Y, X) or (C, Y, X), got shape {arr.shape}")

    idx = list(channel_indices) if channel_indices is not None else list(range(min(3, arr.shape[0])))
    if len(idx) > 3:
        raise LoadError("At most three channels can be rendered as RGB")
    for i in idx:
        if not 0 <= i < arr.shape[0]:
            raise LoadError(f"channel index {i} out of range for {arr.shape[0]} channels")

    planes = [to_display_uint8(arr[i]) if normalize else arr[i].astype(np.uint8) for i in idx]
    while len(planes) < 3:
        planes.append(np.zeros_like(planes[0]))
    return np.stack(planes[:3], axis=-1)
