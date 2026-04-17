# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is puia

puia is a Python library for ML-based volcanic eruption forecasting. It transforms raw seismic time series into tsfresh feature matrices, trains ensembles of classifiers (DT, RF, SVM, etc.), and produces probabilistic eruption forecasts via consensus voting. It supports single-volcano, multi-volcano, and transfer-learning workflows.

## Running tests

```bash
# Built-in test suite
python -c "from puia.tests import run_tests; run_tests()"

# Multi-scale feature tests (requires real data at U:\Research\...\data)
python test_multiscale.py
```

There is no package installer, linter, or CI pipeline. The library is used via direct imports from scripts in the repo root.

## Dependencies

Core: numpy, pandas, scipy, matplotlib, tsfresh, scikit-learn, imblearn, joblib, tqdm, multiprocessing. Optional: obspy (for live seismic data downloads via FDSN).

## Architecture

### Data flow

```
Raw seismic CSV (10-min intervals) → SeismicData.df
    → Transforms applied (zsc2, log, diff, inv, stft variants)
    → ForecastModel._construct_windows() — overlapping sliding windows
    → tsfresh.extract_features() per window → feature matrix (fM)
    → ForecastModel._get_label() — binary: 1 if eruption within look_forward days
    → ForecastModel.train() — ensemble of Ncl classifiers with RandomUnderSampler + FeatureSelector
    → ForecastModel.forecast() — consensus vote across ensemble → probability [0,1]
    → Forecast / AlertModel — performance metrics, ROC, plots
```

### Key modules in `puia/`

- **`data.py`** — `SeismicData` loads `{station}_seismic_data.csv` and `{station}_eruptive_periods.txt` from data_dir. `get_data(ti, tf)` returns a DataFrame slice. Transforms are applied lazily via `_compute_transforms()`.

- **`features.py`** — Three layers:
  - `Feature` (line ~807): Core feature extraction engine. Owns windowing (`_construct_windows`), tsfresh extraction (`_extract_features`), labeling (`_get_label`), and per-year caching. Supports multi-resolution scales via `scales` parameter.
  - `FeaturesSta` (line ~93): Single-station feature matrix loader. Loads cached `.pkl` feature files, applies normalization and reduction.
  - `FeaturesMulti` (line ~356): Multi-station wrapper that concatenates `FeaturesSta` instances.

- **`model.py`** — Three model classes:
  - `ForecastModel` (line ~362): Main class. `train()` builds `Ncl` classifiers (each with random undersampling + feature selection). `forecast()` produces consensus predictions. `hires_forecast()` creates high-resolution forecasts by year subdivision.
  - `MultiVolcanoForecastModel` (line ~1147): Accepts `data={'WIZ': [ti,tf], 'FWVZ': [ti,tf]}` dict. Trains on pooled multi-station data.
  - `CombinedModel` (line ~1256): Legacy combined training interface.

- **`forecast.py`** — `Forecast` container, `AlertModel` (binary alert windows from consensus threshold), `ROC`/`MultiVolcanoROC` (performance curves), `FSS` (Forecast Skill Score).

- **`transforms.py`** — Stateless transform functions: `zsc2` (z-score with rolling min), `log`, `diff`, `inv`, and combinations. Transform names are parsed from data_streams strings (e.g., `'zsc2_rsam'` applies `zsc2` to `rsam` column).

### Feature caching convention

Feature matrices are cached as pickle files in the feature directory:
- Legacy: `fm_{window:.2f}0w_{datastream}_{station}_{year}.pkl`
- Multi-scale: `fm_{window:.2f}w_s{scale_idx}_{datastream}_{station}_{year}.pkl`

Features are extracted per-year and concatenated. Subsequent runs load from cache and only compute missing windows.

### Multi-resolution scales

The `scales` parameter accepts a list of `(window_days, resample_minutes)` tuples:
```python
scales=[(2, 10), (14, 60), (60, 360)]  # fine/mid/coarse
```
Each scale resamples raw data to coarser resolution before windowing and tsfresh extraction. Features are prefixed (`s0__`, `s1__`, etc.) and concatenated horizontally. Coarser scales are forward-filled to align with the finest scale's time index. `scales=None` preserves legacy single-scale behavior.

## External data paths

Real seismic data and pre-computed features live outside this repo:
- Data CSVs + eruption catalogs: `U:\Research\EruptionForecasting\eruptions\data`
- Cached feature matrices: `U:\Research\EruptionForecasting\eruptions\features`

These paths are passed via `data_dir` and `feature_dir` parameters to model constructors.

## Common model usage pattern

```python
from puia.model import ForecastModel, MultiVolcanoForecastModel

fm = ForecastModel(
    window=2., overlap=0.75, look_forward=2.,
    data='WIZ', root='my_run',
    data_streams=['zsc2_rsam', 'zsc2_dsarF'],
    data_dir=DATA_DIR, feature_dir=FEAT_DIR,
    scales=None  # or [(2, 10), (14, 60)]
)
fm.train(ti='2012-01-01', tf='2019-01-01', Ncl=300, Nfts=20, classifier='DT',
         drop_features=['linear_trend_timewise', 'agg_linear_trend'])
forecast = fm.forecast(ti=datetimeify('2019-12-01'), tf=datetimeify('2019-12-15'))
```

Note: `forecast()` requires datetime objects (use `datetimeify()`), while `train()` accepts strings.

## Important conventions

- The `drop_features` list should always include `'linear_trend_timewise'` and `'agg_linear_trend'` — these tsfresh features cause numerical issues.
- Transform-data stream names follow the pattern `{transform}_{band}{suffix}` where suffix `F` indicates a filtered variant (e.g., `zsc2_dsarF`).
- Eruption labeling is binary: a window is labeled 1 if any eruption falls within `look_forward` days after the window end.
- The `root` parameter determines subdirectory names under `models/` and `forecasts/`.
