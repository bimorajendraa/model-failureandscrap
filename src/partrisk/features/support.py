"""Dukungan historis tipe PART - diekstrak dari `feature_builder.py`
(Fase B2 restrukturisasi), logic TIDAK diubah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_support(observations: pd.DataFrame) -> pd.Series:
    """Jumlah observasi tipe PART yang sama sampai titik waktu masing-masing.

    Dihitung point-in-time supaya tidak memakai informasi masa depan. Baris
    dengan observation_on identik sengaja mendapat nilai yang sama (bukan
    diurutkan sembarang), supaya hasilnya deterministik antar-run.
    """
    times = observations["observation_on"].to_numpy("datetime64[ns]")
    support = np.zeros(len(observations), dtype="int64")
    grouped = observations.groupby("item_model_code_clean", sort=False, dropna=False)
    for rows in grouped.indices.values():
        support[rows] = np.searchsorted(np.sort(times[rows]), times[rows], side="right")
    return pd.Series(support, index=observations.index)


def support_totals(observations: pd.DataFrame) -> dict[str, int]:
    """Dukungan akhir per tipe PART, untuk dibekukan ke dalam metadata model."""
    totals = observations.groupby("item_model_code_clean").size()
    return {str(model): int(count) for model, count in totals.items()}


def part_model_support(raw: pd.DataFrame, support_by_model: dict[str, int]) -> pd.Series:
    """Dukungan historis tipe PART untuk jalur prediction.

    Memakai angka yang DIBEKUKAN saat training (tersimpan di metadata), bukan
    dihitung ulang dari data terbaru. Alasannya konsistensi: kategori yang
    dikenal model adalah kategori pada saat model dilatih. Kalau sebuah tipe
    PART baru melewati ambang dukungan di antara dua kali training, menghitung
    ulang akan memunculkan kategori yang belum pernah dilihat model. Angka ini
    otomatis ikut diperbarui setiap kali `train.py` dijalankan.
    """
    return (
        raw["item_model_code_clean"]
        .map(support_by_model)
        .fillna(0)
        .astype("int64")
    )
