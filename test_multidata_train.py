"""
End-to-end training test for multi-source + multi-scale eruption forecasting.
Uses COPZ (Copahue) with seismic + gas data.
Trains on 2018-2021 (gas+seismic overlap period), forecasts on a known eruption.
"""

import sys, os
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = r'U:\Research\EruptionForecasting\eruptions\data'
FEAT_DIR = r'U:\Research\EruptionForecasting\eruptions\features'

from puia.model import ForecastModel
from puia.utilities import datetimeify

def main():
    # ============================================================
    # Test A: Single-scale, seismic+gas, small training run
    # ============================================================
    print("\n" + "="*60)
    print("Test A: Single-scale multi-source (seismic+gas) training")
    print("="*60)

    fm = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='COPZ', root='test_copz_multidata_single',
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        data_sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
        },
        scales=None
    )
    fm.n_jobs = 0  # serial for Windows compatibility

    print(f"Data type: {type(fm.data).__name__}")
    print(f"Data streams: {fm.data_streams}")
    print(f"Data range: {fm.data.ti} to {fm.data.tf}")
    print(f"Eruptions: {len(fm.data.tes)}")
    print(f"Scales: {fm.scales}")

    # Train on gas+seismic overlap period (2018-2020)
    print("\nTraining (Ncl=10, small test)...")
    try:
        fm.train(
            ti='2018-06-01', tf='2020-12-31',
            Ncl=10, Nfts=10, classifier='DT',
            drop_features=['linear_trend_timewise', 'agg_linear_trend'],
            n_jobs=1
        )
        print("Training PASSED")

        # Forecast around the 2020-06-16 eruption
        print("\nForecasting around 2020-06-16 eruption...")
        forecast = fm.forecast(
            ti=datetimeify('2020-06-10'),
            tf=datetimeify('2020-06-20'),
            n_jobs=1
        )
        print(f"Forecast shape: {forecast.shape}")
        print(f"Forecast max consensus: {forecast.max().values}")
        print("Forecast PASSED")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # Test B: Multi-scale, seismic+gas
    # ============================================================
    print("\n" + "="*60)
    print("Test B: Multi-scale multi-source (seismic+gas) training")
    print("="*60)

    fm2 = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='COPZ', root='test_copz_multidata_multiscale',
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        data_sources={
            'seismic': ['rsam', 'dsar'],
            'gas': ['flux'],
        },
        scales=[2, 14]
    )
    fm2.n_jobs = 0

    print(f"Data streams: {fm2.data_streams}")
    print(f"Scales: {[(s['window'], s['resample_min']) for s in fm2.ft.scales]}")

    print("\nTraining (Ncl=10, small test)...")
    try:
        fm2.train(
            ti='2018-06-01', tf='2020-12-31',
            Ncl=10, Nfts=10, classifier='DT',
            drop_features=['linear_trend_timewise', 'agg_linear_trend'],
            n_jobs=1
        )
        print("Multi-scale training PASSED")

        print("\nForecasting around 2020-06-16 eruption...")
        forecast2 = fm2.forecast(
            ti=datetimeify('2020-06-10'),
            tf=datetimeify('2020-06-20'),
            n_jobs=1
        )
        print(f"Forecast shape: {forecast2.shape}")
        print(f"Forecast max consensus: {forecast2.max().values}")
        print("Multi-scale forecast PASSED")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("END-TO-END TESTS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
