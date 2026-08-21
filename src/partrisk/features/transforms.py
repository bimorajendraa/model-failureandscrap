"""Transformasi angka kecil dipakai lintas modul fitur - diekstrak dari
`feature_builder.py` (Fase B2 restrukturisasi) supaya `observations.py`/
`fleet.py`/`failure.py` tidak perlu saling impor untuk dua fungsi kecil ini.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from partrisk import config


def _log1p(values: pd.Series) -> pd.Series:
    """LN(1+x) dengan nilai kosong diperlakukan sebagai 0.

    Kosong di sini berarti "belum pernah terjadi" (mis. belum pernah ada
    corrective), bukan data hilang - itulah kenapa dipasangkan dengan kolom
    penanda has_* supaya model bisa membedakan keduanya.
    """
    return np.log1p(pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0))


def _age_band(days: pd.Series) -> pd.Series:
    index = np.searchsorted(
        config.AGE_BAND_THRESHOLDS,
        pd.to_numeric(days, errors="coerce").fillna(0.0).to_numpy(),
        side="right",
    )
    return pd.Series(np.asarray(config.AGE_BAND_LABELS)[index], index=days.index)
