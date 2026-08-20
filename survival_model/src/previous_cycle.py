"""Audit `previous_cycle_lifetime_mean` (dari `data_reader.get_cycles()` SQL).

Kolom itu TERBUKTI mencampur rata-rata durasi siklus SEBELUMNYA milik item
yang sama APAPUN cara ia berakhir (FAILURE, RIGHT_CENSORED_AT_DATA_END,
REINSTALL_WITHOUT_RECORDED_FAILURE) - bukan "lifetime sampai gagal" seperti
namanya menyiratkan. Modul ini menghitung varian yang lebih jujur secara
point-in-time, dari `cycles` PENUH (populasi yang SAMA dipakai
`previous_cycle_lifetime_mean` di SQL - tidak dibatasi cohort eligible
survival), per item, hanya memakai siklus SEBELUM siklus berjalan (strictly
prior installation_sequence).

Tiga kolom baru:
- `previous_cycle_confirmed_failure_lifetime_mean`: rata-rata durasi siklus
  SEBELUMNYA yang BENAR-BENAR berakhir FAILURE saja.
- `last_confirmed_failure_lifetime`: durasi siklus FAILURE PALING BARU di
  antara siklus-siklus sebelumnya (bukan rata-rata).
- `previous_cycle_end_reason`: cara siklus TEPAT sebelumnya berakhir
  (FAILURE/RIGHT_CENSORED_AT_DATA_END/REINSTALL_WITHOUT_RECORDED_FAILURE/
  NONE_FIRST_CYCLE kalau tidak ada siklus sebelumnya) - fitur kategorikal
  opsional, dipakai HANYA kalau terbukti membantu validation (lihat
  reports/previous_cycle_audit.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NONE_FIRST_CYCLE = "NONE_FIRST_CYCLE"


def audit_previous_cycle_features(cycles: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan `cycles` + 3 kolom audit di atas, satu baris per
    installation_cycle_id (index/urutan asli dipertahankan)."""
    frame = cycles.reset_index(drop=True).copy()
    frame["_sequence"] = (
        frame["installation_cycle_id"].str.rsplit(":", n=1).str[-1].astype(int)
    )
    frame = frame.sort_values(["item_identifier_clean", "_sequence"], kind="stable")

    duration_days = (frame["cycle_end_on"] - frame["installed_on"]) / np.timedelta64(1, "D")
    frame["_failure_duration"] = np.where(frame["cycle_end_reason"].eq("FAILURE"), duration_days, np.nan)

    # shift(1) dulu (nilai siklus TEPAT sebelumnya, batas grup dijaga
    # groupby) - baru expanding/ffill pada hasil shift itu, supaya baris
    # saat ini TIDAK PERNAH ikut menghitung dirinya sendiri. Diverifikasi
    # dengan unit test manual sebelum dipakai di sini.
    frame["_shifted_failure_duration"] = (
        frame.groupby("item_identifier_clean", sort=False)["_failure_duration"].shift(1)
    )
    frame["previous_cycle_confirmed_failure_lifetime_mean"] = (
        frame.groupby("item_identifier_clean", sort=False)["_shifted_failure_duration"]
        .expanding().mean().reset_index(level=0, drop=True)
    )
    frame["last_confirmed_failure_lifetime"] = (
        frame.groupby("item_identifier_clean", sort=False)["_shifted_failure_duration"].ffill()
    )
    frame["previous_cycle_end_reason"] = (
        frame.groupby("item_identifier_clean", sort=False)["cycle_end_reason"]
        .shift(1).fillna(NONE_FIRST_CYCLE)
    )

    frame = frame.drop(columns=["_sequence", "_failure_duration", "_shifted_failure_duration"])
    return frame.sort_index()


def transform_for_model(observations: pd.DataFrame) -> pd.DataFrame:
    """LN(1+x) + penanda has_* untuk 2 kolom numerik audit - pola yang SAMA
    dipakai feature_builder._log1p()/has_previous_cycle untuk
    previous_cycle_lifetime_mean asli (durasi sangat right-skewed)."""
    out = pd.DataFrame(index=observations.index)
    for column in ("previous_cycle_confirmed_failure_lifetime_mean", "last_confirmed_failure_lifetime"):
        values = pd.to_numeric(observations[column], errors="coerce")
        out[f"log_{column}"] = np.log1p(values.fillna(0.0).clip(lower=0.0))
        out[f"has_{column}"] = values.notna()
    return out

