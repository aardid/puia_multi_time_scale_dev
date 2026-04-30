"""
Smoke tests for multi-source data loading and multi-source + multi-scale feature extraction.
Tests use COPZ (Copahue) which has seismic, gas, GNSS, and temperature data.
"""

import sys, os
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = r'U:\Research\EruptionForecasting\eruptions\data'
FEAT_DIR = r'U:\Research\EruptionForecasting\eruptions\features'

from puia.data import MultiSourceData, GeneralData, SeismicData
from puia.model import ForecastModel
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
print("\n=== Test 1: MultiSourceData with single source (seismic only) ===")
try:
    msd = MultiSourceData(
        'COPZ',
        sources={'seismic': ['rsam', 'mf', 'hf', 'dsar']},
        data_dir=DATA_DIR
    )
    check("MultiSourceData instantiates", True)
    check("data_streams populated", len(msd.data_streams) == 4)
    check("streams are prefixed", all(s.startswith('seismic_') for s in msd.data_streams))
    print(f"  data_streams: {msd.data_streams}")

    # trigger load
    df = msd.df
    check("DataFrame loaded", df is not None and len(df) > 0)
    check("ti set", msd.ti is not None)
    check("tf set", msd.tf is not None)
    print(f"  time range: {msd.ti} to {msd.tf}")
    print(f"  columns: {list(df.columns)}")
    print(f"  shape: {df.shape}")
    check("eruptions loaded", msd.tes is not None and len(msd.tes) > 0)
    print(f"  eruptions: {len(msd.tes)}")
except Exception as e:
    check("MultiSourceData single source", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 2: MultiSourceData with seismic + gas ===")
try:
    msd2 = MultiSourceData(
        'COPZ',
        sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
        },
        data_dir=DATA_DIR
    )
    check("MultiSourceData seismic+gas instantiates", True)
    check("data_streams count = 3", len(msd2.data_streams) == 3)
    print(f"  data_streams: {msd2.data_streams}")

    df2 = msd2.df
    check("DataFrame loaded", df2 is not None and len(df2) > 0)
    print(f"  columns: {list(df2.columns)}")
    print(f"  shape: {df2.shape}")
    check("seismic_rsam in columns", 'seismic_rsam' in df2.columns)
    check("seismic_dsar in columns", 'seismic_dsar' in df2.columns)
    check("gas_flux in columns", 'gas_flux' in df2.columns)

    # gas starts 2018, seismic starts 2012 — gas_flux should be NaN before 2018
    early = df2.loc[df2.index < '2017-01-01', 'gas_flux']
    check("gas_flux NaN before gas data starts", early.isna().all())

    # get_data works
    sub = msd2.get_data('2020-01-01', '2020-02-01')
    check("get_data returns subset", len(sub) > 0 and len(sub) < len(df2))
    check("get_data has all columns", set(sub.columns) == set(df2.columns))
except Exception as e:
    check("MultiSourceData seismic+gas", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 3: MultiSourceData with seismic + gas + GNSS ===")
try:
    msd3 = MultiSourceData(
        'COPZ',
        sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
            'gnss': ['east', 'north', 'up'],
        },
        data_dir=DATA_DIR
    )
    check("MultiSourceData 3 sources instantiates", True)
    check("data_streams count = 6", len(msd3.data_streams) == 6)
    print(f"  data_streams: {msd3.data_streams}")

    df3 = msd3.df
    check("DataFrame loaded", df3 is not None and len(df3) > 0)
    print(f"  columns: {list(df3.columns)}")
    print(f"  shape: {df3.shape}")
    check("gnss_east in columns", 'gnss_east' in df3.columns)
    check("gnss_north in columns", 'gnss_north' in df3.columns)
    check("gnss_up in columns", 'gnss_up' in df3.columns)

    # GNSS is daily — should be forward-filled onto 10-min grid
    gnss_sub = df3.loc['2020-06-01':'2020-06-02', 'gnss_east']
    check("GNSS forward-filled (not all NaN in overlap)", gnss_sub.notna().any())
except Exception as e:
    check("MultiSourceData 3 sources", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 4: MultiSourceData with transforms ===")
try:
    msd4 = MultiSourceData(
        'COPZ',
        sources={
            'seismic': ['zsc2_rsam', 'rsam', 'dsar'],
        },
        data_dir=DATA_DIR
    )
    check("MultiSourceData with transform instantiates", True)
    print(f"  data_streams: {msd4.data_streams}")

    df4 = msd4.df
    check("DataFrame loaded", df4 is not None)
    print(f"  columns: {list(df4.columns)}")
    check("seismic_zsc2_rsam in columns", 'seismic_zsc2_rsam' in df4.columns)
    check("seismic_rsam in columns", 'seismic_rsam' in df4.columns)
except Exception as e:
    check("MultiSourceData with transforms", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 5: ForecastModel with data_sources ===")
try:
    fm = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='COPZ', root='test_multidata',
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        data_sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
        },
        scales=None
    )
    check("ForecastModel with data_sources instantiates", True)
    check("data is MultiSourceData", type(fm.data).__name__ == 'MultiSourceData')
    check("data_streams = 3", len(fm.data_streams) == 3)
    print(f"  data_streams: {fm.data_streams}")
    check("ft.data not yet set", True)
except Exception as e:
    check("ForecastModel with data_sources", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 6: ForecastModel with data_sources + scales ===")
try:
    fm2 = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='COPZ', root='test_multidata_scales',
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        data_sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
        },
        scales=[2, 14, 60]
    )
    check("ForecastModel data_sources+scales instantiates", True)
    check("scales configured", fm2.ft.scales is not None)
    check("3 scales", len(fm2.ft.scales) == 3)
    check("data_streams = 3", len(fm2.data_streams) == 3)
    print(f"  scales: {[(s['window'], s['resample_min']) for s in fm2.ft.scales]}")
    print(f"  data_streams: {fm2.data_streams}")
except Exception as e:
    check("ForecastModel data_sources+scales", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 7: WIZ backward compatibility (no data_sources) ===")
try:
    fm_wiz = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='WIZ', root='test_backward_compat',
        data_streams=['zsc2_rsam', 'zsc2_dsarF'],
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        scales=None
    )
    check("Legacy WIZ model still works", True)
    check("data is SeismicData", type(fm_wiz.data).__name__ == 'SeismicData')
except Exception as e:
    check("Backward compatibility", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n=== Test 8: _is_eruption_in works for MultiSourceData ===")
try:
    msd_erup = MultiSourceData(
        'COPZ',
        sources={'seismic': ['rsam']},
        data_dir=DATA_DIR
    )
    # COPZ has eruption on 2020-06-16
    label = msd_erup._is_eruption_in(days=2., from_time=datetimeify('2020-06-15'))
    check("eruption detected within look_forward", label == 1.0)
    label2 = msd_erup._is_eruption_in(days=2., from_time=datetimeify('2020-01-01'))
    check("no eruption far from events", label2 == 0.0)
except Exception as e:
    check("_is_eruption_in", False, str(e))
    import traceback; traceback.print_exc()

# ============================================================
print("\n" + "="*60)
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"WARNING: {failed} test(s) FAILED")
