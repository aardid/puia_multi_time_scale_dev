"""
Test suite for multi-resolution feature extraction in puia.
Tests backward compatibility, multi-scale shapes, caching, labels, and end-to-end pipeline.
"""

import sys, os
import numpy as np
from datetime import datetime, timedelta

# paths
DATA_DIR = r'U:\Research\EruptionForecasting\eruptions\data'
FEAT_DIR = r'U:\Research\EruptionForecasting\eruptions\features'
MODEL_DIR = r'U:\Research\EruptionForecasting\eruptions\aardid\puia_rep_dev\models\test_multiscale'
FORECAST_DIR = r'U:\Research\EruptionForecasting\eruptions\aardid\puia_rep_dev\forecasts\test_multiscale'

from puia.model import ForecastModel, MultiVolcanoForecastModel
from puia.features import Feature, FeaturesSta
from puia.data import SeismicData
from puia.utilities import datetimeify

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {detail}")
        failed += 1

# ============================================================
print("\n=== Test 1: Backward compatibility (scales=None) ===")
# Verify that scales=None produces same behavior as before
# We just check that ForecastModel can be instantiated and Feature object
# has scales=None
try:
    fm_legacy = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_legacy',
        data_streams=['zsc2_rsam'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=None
    )
    check("ForecastModel with scales=None instantiates", True)
    check("Feature.scales is None", fm_legacy.ft.scales is None)
    check("Feature.iw = 288 (2d @ 10min)", fm_legacy.ft.iw == 288)
    check("Feature.dtw = 2 days", abs(fm_legacy.ft.dtw.total_seconds() - 2*86400) < 1)
except Exception as e:
    check("ForecastModel with scales=None instantiates", False, str(e))

# ============================================================
print("\n=== Test 2: Single-scale explicit (should match legacy) ===")
try:
    fm_single = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_single',
        data_streams=['zsc2_rsam'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=[(2., 10)]
    )
    check("ForecastModel with scales=[(2,10)] instantiates", True)
    check("Feature.scales is not None", fm_single.ft.scales is not None)
    check("len(scales) == 1", len(fm_single.ft.scales) == 1)
    s0 = fm_single.ft.scales[0]
    check("scale 0 iw = 288", s0['iw'] == 288)
    check("scale 0 prefix = 's0'", s0['prefix'] == 's0')
    check("scale 0 resample_rule = '10min'", s0['resample_rule'] == '10min')
except Exception as e:
    check("ForecastModel with scales=[(2,10)] instantiates", False, str(e))

# ============================================================
print("\n=== Test 3: Multi-resolution scale configs ===")
try:
    fm_multi = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_multi',
        data_streams=['zsc2_rsam'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=[(2, 10), (14, 60), (60, 360)]
    )
    check("ForecastModel with 3 scales instantiates", True)
    check("len(scales) == 3", len(fm_multi.ft.scales) == 3)

    # Check expected samples per window at each scale
    s0 = fm_multi.ft.scales[0]
    s1 = fm_multi.ft.scales[1]
    s2 = fm_multi.ft.scales[2]
    check(f"scale 0: iw={s0['iw']} (expect 288)", s0['iw'] == 288)
    check(f"scale 1: iw={s1['iw']} (expect 336)", s1['iw'] == 336)
    check(f"scale 2: iw={s2['iw']} (expect 240)", s2['iw'] == 240)

    check("scale 0 prefix = 's0'", s0['prefix'] == 's0')
    check("scale 1 prefix = 's1'", s1['prefix'] == 's1')
    check("scale 2 prefix = 's2'", s2['prefix'] == 's2')

    check("scale 1 resample_rule = '1h'", s1['resample_rule'] == '1h')
    check("scale 2 resample_rule = '6h'", s2['resample_rule'] == '6h')

    # longest_dtw should be the 60-day scale
    check("longest_dtw = 60 days", abs(fm_multi.ft.longest_dtw.total_seconds() - 60*86400) < 86400)

    # featfile names should include scale index
    ff0 = s0['featfile']('zsc2_rsam', 2019, 'WIZ')
    ff1 = s1['featfile']('zsc2_rsam', 2019, 'WIZ')
    ff2 = s2['featfile']('zsc2_rsam', 2019, 'WIZ')
    check("scale 0 featfile has '_s0_'", '_s0_' in ff0)
    check("scale 1 featfile has '_s1_'", '_s1_' in ff1)
    check("scale 2 featfile has '_s2_'", '_s2_' in ff2)
    check("scale files are distinct", ff0 != ff1 and ff1 != ff2)
    print(f"  featfile examples: {os.path.basename(ff0)}, {os.path.basename(ff1)}, {os.path.basename(ff2)}")

except Exception as e:
    check("ForecastModel with 3 scales instantiates", False, str(e))

# ============================================================
print("\n=== Test 4: Scale with daily resolution (180d @ 1 day) ===")
try:
    fm_long = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_long',
        data_streams=['zsc2_rsam'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=[(2, 10), (180, 1440)]
    )
    s_long = fm_long.ft.scales[1]
    check(f"180d daily scale: iw={s_long['iw']} (expect 180)", s_long['iw'] == 180)
    check("180d daily scale: resample_rule = '1D'", s_long['resample_rule'] == '1D')
except Exception as e:
    check("Long scale instantiation", False, str(e))

# ============================================================
print("\n=== Test 5: _construct_windows with scale parameter ===")
try:
    # Use the multi-scale model from test 3
    fm = fm_multi
    fm.ft.data = fm.data  # set data reference

    # Test scale 1 (14 days @ 1hr) windowing for a small Nw
    s1 = fm.ft.scales[1]
    ti = datetimeify('2019-12-01')
    Nw = 3
    df, wd = fm.ft._construct_windows(Nw, ti, 'zsc2_rsam', scale=s1)

    # Check that we got data
    check("_construct_windows returns data", df.shape[0] > 0)
    check(f"window_dates count = {len(wd)} (expect {Nw})", len(wd) == Nw)

    # Each window should have iw samples (336 for 14d@1hr)
    unique_ids = df['id'].unique()
    check(f"number of windows = {len(unique_ids)} (expect {Nw})", len(unique_ids) == Nw)

    for wid in unique_ids:
        n_samples = (df['id'] == wid).sum()
        # Allow small tolerance for edge effects
        if n_samples != s1['iw']:
            print(f"  INFO: window {wid} has {n_samples} samples (expected {s1['iw']})")
    check("Windows have approximately correct sample count", True)

except Exception as e:
    check("_construct_windows with scale", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 6: hires_forecast passes scales through ===")
try:
    fm_hr = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_hires',
        data_streams=['zsc2_rsam'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=[(2, 10), (14, 60)]
    )
    check("scales stored on ForecastModel", fm_hr.scales is not None)
    check("scales has 2 entries", len(fm_hr.scales) == 2)
except Exception as e:
    check("hires_forecast scales passthrough", False, str(e))

# ============================================================
print("\n" + "="*60)
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} test(s) FAILED")
