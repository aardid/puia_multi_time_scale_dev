# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is puia

puia is a Python library for ML-based volcanic eruption forecasting. It transforms raw time series (seismic, gas, GNSS, temperature, etc.) into tsfresh feature matrices, trains ensembles of classifiers (DT, RF, SVM, etc.), and produces probabilistic eruption forecasts via consensus voting. It supports single-volcano, multi-volcano, multi-source data, multi-resolution scales, and transfer-learning workflows.

## Running tests

```bash
# Built-in test suite
python -c "from puia.tests import run_tests; run_tests()"

# Multi-scale feature tests (requires real data at U:\Research\...\data)
python test_multiscale.py

# Multi-source data smoke tests (39 tests: data loading, merging, transforms, backward compat)
python test_multidata.py

# Multi-source + multi-scale end-to-end training/forecast (requires COPZ data)
python test_multidata_train.py
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

- **`data.py`** — Data loading classes:
  - `SeismicData`: loads `{station}_seismic_data.csv` and `{station}_eruptive_periods.txt` from data_dir. `get_data(ti, tf)` returns a DataFrame slice. Transforms are applied lazily via `_compute_transforms()`.
  - `MultiSourceData`: loads multiple `{station}_{source_type}_data.csv` files (seismic, gas, gnss, temperature, etc.), merges them into a single DataFrame with `{source_type}_{column}` prefixed columns. Coarser data (e.g., daily GNSS) is forward-filled onto the finest time grid (e.g., 10-min seismic). Supports transforms. Used via `data_sources` parameter on `ForecastModel`.

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

The `scales` parameter enables multi-timescale feature extraction. It accepts three formats:

```python
# 1. List of days (simplest) — resampling auto-computed to keep ~288 samples per window
scales = [2, 14, 60, 180]

# 2. List of (window_days, resample_minutes) tuples — full manual control
scales = [(2, 10), (14, 60), (60, 360)]

# 3. Mixed — numbers and tuples can be combined
scales = [2, (14, 60), 60]
```

When a scale is given as a single number, the resampling interval is chosen automatically to keep approximately 288 samples per window (matching the base 2-day @ 10-min scale). The ratio is `resample_min = window_days * 5`, rounded to a clean interval (nearest 10min, 1hr, or 1day).

Each scale resamples raw data to coarser resolution before windowing and tsfresh extraction. Features are prefixed (`s0__`, `s1__`, etc.) and concatenated horizontally. Coarser scales are forward-filled to align with the finest scale's time index. `scales=None` preserves legacy single-scale behavior.

## External data paths

Real seismic data and pre-computed features live outside this repo:
- Data CSVs + eruption catalogs: `U:\Research\EruptionForecasting\eruptions\data`
- Cached feature matrices: `U:\Research\EruptionForecasting\eruptions\features`

These paths are passed via `data_dir` and `feature_dir` parameters to model constructors.

## Common model usage patterns

### Single-source (legacy seismic-only)

```python
from puia.model import ForecastModel
fm = ForecastModel(
    window=2., overlap=0.75, look_forward=2.,
    data='WIZ', root='my_run',
    data_streams=['zsc2_rsam', 'zsc2_dsarF'],
    data_dir=DATA_DIR, feature_dir=FEAT_DIR,
    scales=None  # or [2, 14, 60] or [(2, 10), (14, 60)]
)
fm.train(ti='2012-01-01', tf='2019-01-01', Ncl=300, Nfts=20, classifier='DT',
         drop_features=['linear_trend_timewise', 'agg_linear_trend'])
forecast = fm.forecast(ti=datetimeify('2019-12-01'), tf=datetimeify('2019-12-15'))
```

### Multi-source (seismic + gas + GNSS, etc.)

```python
from puia.model import ForecastModel
fm = ForecastModel(
    window=2., overlap=0.75, look_forward=2.,
    data='COPZ', root='copz_multisource',
    data_dir=DATA_DIR, feature_dir=FEAT_DIR,
    data_sources={
        'seismic': ['rsam', 'dsar'],
        'gas': ['flux'],
        'gnss': ['east', 'north', 'up'],  # optional
    },
    scales=[2, 14]  # optional multi-resolution
)
fm.train(ti='2018-06-01', tf='2020-12-31', Ncl=300, Nfts=20, classifier='DT',
         drop_features=['linear_trend_timewise', 'agg_linear_trend'])
forecast = fm.forecast(ti=datetimeify('2020-06-10'), tf=datetimeify('2020-06-20'))
```

The `data_sources` parameter replaces `data_streams` when using multiple source types. Each source type maps to a CSV file `{station}_{source_type}_data.csv` in the data directory. When `data_sources` is provided, `data_streams` is auto-generated with prefixed names (e.g., `seismic_rsam`, `gas_flux`).

Note: `forecast()` requires datetime objects (use `datetimeify()`), while `train()` accepts strings.

## Important conventions

- The `drop_features` list should always include `'linear_trend_timewise'` and `'agg_linear_trend'` — these tsfresh features cause numerical issues.
- Transform-data stream names follow the pattern `{transform}_{band}{suffix}` where suffix `F` indicates a filtered variant (e.g., `zsc2_dsarF`).
- Eruption labeling is binary: a window is labeled 1 if any eruption falls within `look_forward` days after the window end.
- The `root` parameter determines subdirectory names under `models/` and `forecasts/`.
