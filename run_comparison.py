"""
Performance comparison: seismic-only vs multi-source eruption forecasting at Copahue.

Train on 2018-06-01 to 2020-12-31 (eruptions: 2019-08, 2020-06 through 2020-09).
Forecast on 2021 (out-of-sample eruptions: 2021-07-02, 2021-08-10).
Three configurations:
  A) Seismic only (rsam, dsar)
  B) Seismic + gas (rsam, dsar, flux)
  C) Seismic + gas + GNSS (rsam, dsar, flux, east, north, up)
All use scales=[2, 14] for multi-resolution.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

DATA_DIR = r'U:\Research\EruptionForecasting\eruptions\data'
FEAT_DIR = r'U:\Research\EruptionForecasting\eruptions\features'

from puia.model import ForecastModel
from puia.utilities import datetimeify

TRAIN_TI = '2018-06-01'
TRAIN_TF = '2020-12-31'
FCST_TI = datetimeify('2021-01-01')
FCST_TF = datetimeify('2021-12-31')
NCL = 100
NFTS = 20
SCALES = [2, 14]
CLASSIFIER = 'DT'
DROP = ['linear_trend_timewise', 'agg_linear_trend']

ERUPTIONS_2021 = [datetime(2021,7,2,12,0,0), datetime(2021,8,10,12,0,0)]

CONFIGS = {
    'A_seismic_only': {
        'data_sources': {'seismic': ['rsam', 'dsar']},
        'label': 'Seismic only',
        'color': 'C0',
    },
    'B_seismic_gas': {
        'data_sources': {'seismic': ['rsam', 'dsar'], 'gas': ['flux']},
        'label': 'Seismic + Gas',
        'color': 'C1',
    },
    'C_seismic_gas_gnss': {
        'data_sources': {'seismic': ['rsam', 'dsar'], 'gas': ['flux'], 'gnss': ['east', 'north', 'up']},
        'label': 'Seismic + Gas + GNSS',
        'color': 'C2',
    },
}


def run_config(name, cfg):
    print(f"\n{'='*60}")
    print(f"Config {name}: {cfg['label']}")
    print(f"{'='*60}")

    fm = ForecastModel(
        window=2., overlap=0.75, look_forward=2.,
        data='COPZ', root=f'comparison_{name}',
        data_dir=DATA_DIR, feature_dir=FEAT_DIR,
        data_sources=cfg['data_sources'],
        scales=SCALES,
    )
    fm.n_jobs = 0

    print(f"Data streams: {fm.data_streams}")
    print(f"Scales: {[(s['window'], s['resample_min']) for s in fm.ft.scales]}")

    print(f"\nTraining (Ncl={NCL})...")
    fm.train(
        ti=TRAIN_TI, tf=TRAIN_TF,
        Ncl=NCL, Nfts=NFTS, classifier=CLASSIFIER,
        drop_features=DROP, n_jobs=1,
    )
    print("Training complete.")

    print(f"\nForecasting {FCST_TI.date()} to {FCST_TF.date()}...")
    forecast = fm.forecast(ti=FCST_TI, tf=FCST_TF, n_jobs=1)
    print(f"Forecast shape: {forecast.shape}")
    print(f"Max consensus: {forecast.max().values}")

    return forecast


def plot_comparison(forecasts):
    fig, axes = plt.subplots(len(forecasts)+1, 1, figsize=(14, 3*(len(forecasts)+1)),
                              sharex=True, gridspec_kw={'hspace': 0.08})

    for i, (name, (cfg, fc)) in enumerate(forecasts.items()):
        ax = axes[i]
        ax.plot(fc.index, fc.iloc[:, 0], color=cfg['color'], linewidth=0.8)
        ax.fill_between(fc.index, 0, fc.iloc[:, 0], alpha=0.3, color=cfg['color'])
        for te in ERUPTIONS_2021:
            ax.axvline(te, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.set_ylabel('Consensus')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(cfg['label'], fontsize=11, loc='left')
        ax.grid(True, alpha=0.3)

    # overlay all on last panel
    ax = axes[-1]
    for name, (cfg, fc) in forecasts.items():
        ax.plot(fc.index, fc.iloc[:, 0], color=cfg['color'], linewidth=0.8, label=cfg['label'])
    for te in ERUPTIONS_2021:
        ax.axvline(te, color='red', linewidth=1.5, linestyle='--', alpha=0.8, label='Eruption' if te == ERUPTIONS_2021[0] else None)
    ax.set_ylabel('Consensus')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('All models', fontsize=11, loc='left')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Date')

    plt.suptitle('COPZ Multi-source Forecast Comparison (2021, out-of-sample)', fontsize=13)
    plt.tight_layout()
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison_forecast_2021.png')
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to {outfile}")
    plt.close()


def print_summary(forecasts):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training: {TRAIN_TI} to {TRAIN_TF}")
    print(f"Forecast: {FCST_TI.date()} to {FCST_TF.date()}")
    print(f"2021 eruptions: {[str(e.date()) for e in ERUPTIONS_2021]}")
    print(f"Ncl={NCL}, Nfts={NFTS}, scales={SCALES}, classifier={CLASSIFIER}")
    print()

    for name, (cfg, fc) in forecasts.items():
        vals = fc.iloc[:, 0]
        # check forecast value near eruptions (within look_forward=2 days before)
        hits = []
        for te in ERUPTIONS_2021:
            window = vals[(vals.index >= te - pd.Timedelta(days=2)) & (vals.index <= te)]
            if len(window) > 0:
                hits.append(window.max())
            else:
                hits.append(np.nan)

        print(f"{cfg['label']:30s}  max={vals.max():.2f}  mean={vals.mean():.3f}  "
              f"eruption hits: {['%.2f'%h for h in hits]}")


def main():
    forecasts = {}
    for name, cfg in CONFIGS.items():
        try:
            fc = run_config(name, cfg)
            forecasts[name] = (cfg, fc)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    if forecasts:
        print_summary(forecasts)
        plot_comparison(forecasts)

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
