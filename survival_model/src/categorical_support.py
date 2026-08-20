"""Dukungan historis kategori, digeneralisasi untuk kolom apa pun.

`feature_builder.cumulative_support()`/`support_totals()` di root HARDCODED
ke kolom `item_model_code_clean` - tidak bisa dipakai langsung untuk kolom
survival baru (`place_at_install`, dst.) tanpa mengubah `feature_builder.py`.
Modul ini menggeneralisasi MEKANISME yang SAMA PERSIS (cumulative point-in-
time count + freeze totals saat training) ke kolom mana pun - bukan logic
baru, hanya parameterisasi. Threshold di sini KHUSUS survival (skala
~15rb lifecycle TRAIN), TIDAK reuse `config.MIN_PART_MODEL_SUPPORT=300`
milik classification (skala 251rb baris) - lihat reports/category_threshold.md
untuk hasil pemilihannya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN_LABEL = "UNKNOWN"
LOW_SUPPORT_LABEL = "LOW_SUPPORT"


def cumulative_support(frame: pd.DataFrame, column: str, time_column: str) -> pd.Series:
    """Jumlah observasi nilai kategori yang SAMA sampai titik waktu masing-
    masing baris (point-in-time, termasuk baris itu sendiri) - identik
    mekanismenya dengan feature_builder.cumulative_support, digeneralisasi
    ke kolom apa pun."""
    times = frame[time_column].to_numpy("datetime64[ns]")
    support = np.zeros(len(frame), dtype="int64")
    grouped = frame.groupby(column, sort=False, dropna=False)
    for rows in grouped.indices.values():
        support[rows] = np.searchsorted(np.sort(times[rows]), times[rows], side="right")
    return pd.Series(support, index=frame.index)


def support_totals(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Dukungan akhir per kategori - dibekukan ke metadata model saat
    training, dipakai ulang saat prediksi (alasan sama seperti
    feature_builder.part_model_support di root: kategori yang dikenal model
    adalah kategori pada saat model dilatih)."""
    totals = frame.groupby(column).size()
    return {str(key): int(count) for key, count in totals.items()}


def apply_threshold(
    values: pd.Series, support: pd.Series, threshold: int,
    *, low_label: str = LOW_SUPPORT_LABEL, unknown_label: str = UNKNOWN_LABEL,
) -> pd.Series:
    """Kelompokkan nilai bersupport < threshold jadi satu kategori bersama,
    dan nilai kosong jadi UNKNOWN - pola yang sama dengan
    part_model_category di feature_builder.build_features()."""
    support_numeric = pd.to_numeric(support, errors="coerce").fillna(0)
    return pd.Series(
        np.where(
            values.isna(),
            unknown_label,
            np.where(support_numeric < threshold, low_label, values.astype(str)),
        ),
        index=values.index,
    )


def apply_frozen_totals(
    values: pd.Series, totals: dict[str, int], threshold: int,
    *, low_label: str = LOW_SUPPORT_LABEL, unknown_label: str = UNKNOWN_LABEL,
) -> pd.Series:
    """Versi apply_threshold() untuk jalur prediction: pakai dukungan yang
    SUDAH DIBEKUKAN saat training (metadata), bukan dihitung ulang - supaya
    kategori yang dikenal model konsisten dengan saat ia dilatih."""
    support = values.map(totals).fillna(0)
    return apply_threshold(values, support, threshold, low_label=low_label, unknown_label=unknown_label)


def threshold_report_stats(
    train_values: pd.Series, val_values: pd.Series, test_values: pd.Series, threshold: int,
) -> dict:
    """Statistik satu baris laporan threshold experiment: jumlah kategori
    asli di TRAIN, berapa yang tergabung LOW_SUPPORT pada threshold ini, dan
    berapa kategori di VALIDATION/TEST yang TIDAK PERNAH terlihat di TRAIN -
    dukungan dibekukan dari TRAIN saja (meniru persis kondisi saat prediksi,
    lihat apply_frozen_totals())."""
    totals = support_totals(pd.DataFrame({"_v": train_values}), "_v")
    original_categories = len(totals)
    merged_categories = sum(1 for count in totals.values() if count < threshold)

    train_known = {name for name, count in totals.items() if count >= threshold}
    val_unseen = int((~val_values.dropna().astype(str).isin(train_known)).sum())
    test_unseen = int((~test_values.dropna().astype(str).isin(train_known)).sum())

    return {
        "threshold": threshold,
        "original_categories": original_categories,
        "merged_into_low_support": merged_categories,
        "kept_as_own_category": original_categories - merged_categories,
        "val_rows_unseen_category": val_unseen,
        "val_rows_total": int(val_values.notna().sum()),
        "test_rows_unseen_category": test_unseen,
        "test_rows_total": int(test_values.notna().sum()),
    }
