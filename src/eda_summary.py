from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import skew, kurtosis

from src.io_npz import load_npz


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    return x[np.isfinite(x)]


def print_eda_summary(npz_path: str, label: str = "DATASET") -> None:
    ds = load_npz(npz_path)

    N = ds.y.shape[0]
    H, W = ds.X.shape[1], ds.X.shape[2]   # NHWC after load_npz
    C = ds.X.shape[-1]
    K = ds.X_extra.shape[1]
    stations = sorted(set(ds.station.tolist()))
    n_stations = len(stations)

    y = _finite(ds.y)

    stats = {
        "n": int(len(y)),
        "mean": float(np.mean(y)) if len(y) else float("nan"),
        "std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "min": float(np.min(y)) if len(y) else float("nan"),
        "max": float(np.max(y)) if len(y) else float("nan"),
        "skew": float(skew(y)) if len(y) > 2 else float("nan"),
        "kurtosis": float(kurtosis(y, fisher=True)) if len(y) > 3 else float("nan"),
    }

    counts = Counter(ds.station.tolist()).most_common()

    print("=" * 80)
    print(f"EDA — {label}")
    print("=" * 80)
    print(f"File: {Path(npz_path).name}")
    print(f"Samples (N): {N}")
    print(f"Patch: {H}x{W}")
    print(f"Bands (C): {C}")
    print(f"AUX vars: {K}")
    print(f"Stations: {n_stations}\n")

    print("Target summary:")
    print(f"{'n':>10}: {stats['n']}")
    print(f"{'mean':>10}: {stats['mean']:.4f}")
    print(f"{'std':>10}: {stats['std']:.5f}")
    print(f"{'min':>10}: {stats['min']:.4f}")
    print(f"{'max':>10}: {stats['max']:.4f}")
    print(f"{'skew':>10}: {stats['skew']:.5f}")
    print(f"{'kurtosis':>10}: {stats['kurtosis']:.5f}\n")

    print("Top station counts:")
    for st, n in counts[:10]:
        print(f"  - {st}: {n}")
    print()


if __name__ == "__main__":
    # works with synthetic sample (public) or a private NPZ placed in data/
    print_eda_summary("data/sample_data.npz", label="SYNTHETIC")
