from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


BAND_ORDER_12 = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]


@dataclass
class NPZDataset:
    """
    Container for a patch-based EO + in-situ dataset.

    Expected keys in NPZ:
      - X: patch tensor (N, C, H, W) or (N, H, W, C)
      - y: target (N,)
      - station: (N,) str
      - date: (optional) (N,) str
      - X_extra: auxiliary vars (N, K)
      - aux_names: (K,) str
      - band_names: (C,) str
    """
    X: np.ndarray
    y: np.ndarray
    station: np.ndarray
    date: Optional[np.ndarray]
    X_extra: np.ndarray
    aux_names: List[str]
    band_names: List[str]


def _as_str_array(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def ensure_nhwc(X: np.ndarray) -> np.ndarray:
    """
    Ensure patch tensor is NHWC.

    If input is NCHW (N, C, H, W) with C=12, converts to NHWC (N, H, W, C).
    If already NHWC, returns as-is.
    """
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"X must be 4D, got {X.shape}")

    # common in EO: (N, 12, 5, 5)
    if X.shape[1] == 12 and X.shape[2] <= 25 and X.shape[3] <= 25:
        X = np.transpose(X, (0, 2, 3, 1))
    return X


def validate_npz_schema(d: dict, name: str = "dataset") -> None:
    required = ["X", "y", "station", "X_extra"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{name}: missing keys in NPZ: {missing}")

    X = d["X"]
    y = d["y"]
    station = d["station"]
    X_extra = d["X_extra"]

    if np.asarray(X).ndim != 4:
        raise ValueError(f"{name}: X must be 4D, got {np.asarray(X).shape}")
    if np.asarray(y).ndim not in (1, 2):
        raise ValueError(f"{name}: y must be 1D/2D, got {np.asarray(y).shape}")
    if np.asarray(X_extra).ndim != 2:
        raise ValueError(f"{name}: X_extra must be 2D, got {np.asarray(X_extra).shape}")

    n = np.asarray(X).shape[0]
    if np.asarray(y).reshape(-1).shape[0] != n:
        raise ValueError(f"{name}: inconsistent lengths: X has N={n}, y has {np.asarray(y).reshape(-1).shape[0]}")
    if np.asarray(station).shape[0] != n:
        raise ValueError(f"{name}: inconsistent lengths: X has N={n}, station has {np.asarray(station).shape[0]}")
    if np.asarray(X_extra).shape[0] != n:
        raise ValueError(f"{name}: inconsistent lengths: X has N={n}, X_extra has {np.asarray(X_extra).shape[0]}")

    if "date" in d and np.asarray(d["date"]).shape[0] != n:
        raise ValueError(f"{name}: inconsistent lengths: date has {np.asarray(d['date']).shape[0]} vs X has {n}")


def load_npz(path: Union[str, Path]) -> NPZDataset:
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    validate_npz_schema(d, name=path.name)

    X = ensure_nhwc(d["X"]).astype(float)
    y = d["y"].astype(float).reshape(-1)
    station = _as_str_array(d["station"])

    date = None
    if "date" in d:
        date = _as_str_array(d["date"])

    X_extra = d["X_extra"].astype(float)

    aux_names = d["aux_names"] if "aux_names" in d else None
    if aux_names is None:
        aux_names = [f"AUX_{i}" for i in range(X_extra.shape[1])]
    else:
        aux_names = [str(a) for a in aux_names]

    band_names = d["band_names"] if "band_names" in d else None
    if band_names is None:
        # fall back to standard Sentinel-2 12-band order if C==12
        C = X.shape[-1]
        band_names = BAND_ORDER_12[:C] if C == 12 else [f"BAND_{i}" for i in range(C)]
    else:
        band_names = [str(b) for b in band_names]

    return NPZDataset(
        X=X, y=y, station=station, date=date,
        X_extra=X_extra, aux_names=aux_names, band_names=band_names
    )


def to_aux_dataframe(ds: NPZDataset, dataset_label: str) -> pd.DataFrame:
    """
    Tidy DataFrame for target + auxiliary variables.
    """
    df = pd.DataFrame(ds.X_extra, columns=ds.aux_names)
    df.insert(0, "target", ds.y)
    df.insert(0, "dataset", dataset_label)
    df.insert(0, "station", ds.station)
    if ds.date is not None:
        df.insert(0, "date", ds.date)
    return df


def extract_patch_features(ds: NPZDataset) -> Dict[str, np.ndarray]:
    """
    From patches compute:
      - center pixel  (N,C)
      - patch mean    (N,C)
      - patch std     (N,C)
    """
    X = ds.X
    h, w = X.shape[1], X.shape[2]
    cy, cx = h // 2, w // 2

    center = X[:, cy, cx, :]
    mean = np.nanmean(X, axis=(1, 2))
    std = np.nanstd(X, axis=(1, 2), ddof=1)

    return {"center": center, "mean": mean, "std": std}
