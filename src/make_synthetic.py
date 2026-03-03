from __future__ import annotations

from pathlib import Path
import numpy as np

from src.io_npz import BAND_ORDER_12


def make_synthetic_npz(
    out_path: str = "data/sample_data.npz",
    n: int = 500,
    h: int = 5,
    w: int = 5,
    c: int = 12,
    k_aux: int = 5,
    seed: int = 42,
) -> Path:
    """
    Create a synthetic NPZ matching the schema of the private dataset.
    This allows the public repo to run end-to-end without sharing real data.
    """
    rng = np.random.default_rng(seed)

    # mimic Sentinel-2 reflectance patches (N, C, H, W) then stored as NCHW (common EO)
    X = rng.normal(loc=0.06, scale=0.02, size=(n, c, h, w)).astype(np.float32)
    X = np.clip(X, 0.0, 0.25)

    # auxiliary variables (example: 5 vars)
    aux_names = ["Temperature", "Salinity", "Dissolved Oxygen", "FDOM", "Turbidity"][:k_aux]
    X_extra = rng.normal(size=(n, k_aux)).astype(np.float32)

    # metadata
    station = np.array(rng.choice(["E1", "S1"], size=n), dtype=object)
    date = np.array([f"2024-{m:02d}-{d:02d}" for m, d in zip(rng.integers(1, 13, n), rng.integers(1, 28, n))], dtype=object)
    band_names = np.array(BAND_ORDER_12[:c], dtype=object)

    # create a target with some dependence on bands + aux so correlations are meaningful
    patch_mean = X.mean(axis=(2, 3))  # (n, c)
    y = (
        40 * patch_mean[:, 2]  # B03-like
        - 25 * patch_mean[:, 4]  # B05-like
        + 3 * X_extra[:, 0]      # Temperature-like
        + rng.normal(scale=1.5, size=n)
    ).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        station=station,
        date=date,
        X_extra=X_extra,
        aux_names=np.array(aux_names, dtype=object),
        band_names=band_names,
    )
    return out_path


if __name__ == "__main__":
    p = make_synthetic_npz()
    print("Saved:", p)
