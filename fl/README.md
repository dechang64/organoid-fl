# ⚠️ LEGACY DIRECTORY

This directory contains the original prototype code. It has been superseded by
the refactored modules in `analysis/`, `data/`, `utils/`, and `visualization/`.

## Migration Map

| Legacy File | Replacement |
|-------------|-------------|
| `federated_learning.py` | `analysis/fl_engine.py` |
| `feature_extractor.py` | `analysis/feature_extractor_v2.py` |
| `generate_data.py` | `data/synthetic.py` |
| `demo.py` | `app.py` + `modules/` |
| `demo_phase3.py` | `app.py` + `modules/` |

## Status

- **DO NOT** modify files here for new features.
- **DO** fix critical bugs here if they affect existing demos.
- New development happens in `analysis/`, `modules/`, `visualization/`.
