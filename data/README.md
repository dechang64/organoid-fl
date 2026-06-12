# Organoid-FL Datasets

## 1. Intestinal Organoid (Downloaded ✅)

- **Source**: Zenodo DOI:10.5281/zenodo.6768583
- **Paper**: Tellu – an object-detector algorithm for automatic classification of intestinal organoids (PMC, 2023)
- **Extended by**: Multi-Class Segmentation and Classification of Intestinal Organoids (MDPI Applied Sciences, 2024)
- **Size**: 840 images (756 train / 84 val), 23,065 annotations
- **Resolution**: 1280×960 RGB JPEG
- **Format**: YOLO (ready for Ultralytics)
- **Classes**:
  - 0: organoid0 (cystic non-budding) — 11,922 instances
  - 1: organoid1 (early organoid) — 5,510 instances
  - 2: organoid3 (late organoid) — 3,367 instances
  - 3: spheroid — 2,266 instances
- **FL relevance**: 4 classes with clear morphological hierarchy, long-tail distribution (5:1 ratio Org0:Sph)
- **Path**: `intestinal_organoid/OrganoidDataset/`

## 2. MultiOrg (Kaggle — needs auth)

- **Source**: Kaggle https://www.kaggle.com/datasets/christinabukas/mutliorg/
- **Paper**: MultiOrg: A Multi-rater Organoid-detection Dataset (NeurIPS 2024)
- **Size**: 411 images (356 train / 55 test), ~60,000 lung organoid annotations
- **Format**: COCO JSON
- **Annotators**: 2 experts × 2 time points = 4 label sets (multi-rater)
- **FL relevance**: Multi-rater → natural Non-IID (each rater = client), detection task
- **Download command** (requires Kaggle API key):
  ```bash
  kaggle datasets download -d christinabukas/mutliorg
  ```

## FL Experiment Design

### Intestinal Organoid — Phase 2 EWA Validation

| Client | Data Split | Expected Specialty |
|--------|-----------|-------------------|
| Client A | 80% Org0 + 5% each other | Specialist in cystic organoids |
| Client B | 80% Org1 + Org3 + 5% each other | Specialist in mature organoids |
| Client C | 40% Spheroid + 20% each other | Generalist, spheroid-heavy |

This creates:
- **Mild Non-IID**: 60/20/20 split
- **Moderate Non-IID**: 80/10/10 split (primary)
- **Extreme Non-IID**: 95/3/2 split

Expected outcome: EWA should outperform FedAvg in Moderate Non-IID by weighting the specialist client higher for its dominant class.

### MultiOrg — Cross-Rater FL

| Client | Data Source | Natural Non-IID Type |
|--------|-----------|---------------------|
| Client A | Annotator A labels | Different boundary decisions |
| Client B | Annotator B labels | Different size thresholds |
| Client C | Consensus labels | "Clean" ground truth |

This tests FL under annotation uncertainty — a real-world scenario in biomedical imaging.

## 3. Organoid Patches (Classification — Generated ✅)

- **Source**: Cropped from Intestinal Organoid detection dataset
- **Script**: `crop_patches.py`
- **Size**: 23,052 patches (18,493 train / 4,559 val), 4 classes
- **Classes**:
  - organoid0 (cystic non-budding): 11,911 patches
  - organoid1 (early organoid): 5,510 patches
  - organoid3 (late organoid): 3,366 patches
  - spheroid: 2,265 patches
- **Format**: Folder-based classification (train/cls_name/xxx.jpg), compatible with Ultralytics
- **Patch sizes**: median 46-95px depending on class, resized to 224×224 for ResNet training
- **Config**: `organoid_patches/data.yaml`
- **Summary**: `organoid_patches/summary.json`

### Comparison: Original vs New Dataset

| | Original (Synthetic) | Organoid Patches (Real) |
|---|---|---|
| Images | 600 | 23,052 |
| Classes | 3 (healthy/early/late) | 4 (org0/org1/org3/sph) |
| Source | Python-generated | Real mouse intestinal organoid microscopy |
| Task | Classification | Classification (cropped from detection) |
| Format | Folder-based | Folder-based |
| Long-tail ratio | ~1:1:1 | 5.3:2.4:1.5:1 |
| FL Non-IID potential | Low (balanced) | High (natural long-tail) |
