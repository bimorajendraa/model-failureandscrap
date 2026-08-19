"""Faktor risiko dalam bahasa manusia, diambil dari fitur yang benar-benar ada.

Ini BUKAN penjelasan kontribusi model. Yang ditampilkan adalah kondisi nyata
PART yang menjadi masukan model - persis kolom yang dihitung
feature_builder.py - supaya angka probabilitas tidak berdiri sendiri tanpa
konteks.

Yang sengaja TIDAK dilakukan:

- tidak mengarang alasan yang tidak ada datanya;
- tidak mengklaim seberapa besar satu faktor menaikkan skor. Untuk itu perlu
  analisis kontribusi per-fitur (SHAP), dan itu pekerjaan tahap berikutnya.

Label `direction` di bawah adalah arah umum fitur tersebut menurut definisinya
(mis. "pernah rusak" jelas bukan pertanda baik), bukan hasil pengukuran pada
model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

DISCLAIMER = (
    "Faktor di bawah adalah kondisi PART yang menjadi masukan model, bukan "
    "kontribusi terukur terhadap skor. Analisis kontribusi per-fitur (SHAP) "
    "belum tersedia."
)

# Hitungan korektif memakai definisi yang sama dengan yang dipelajari model -
# jumlah CATATAN kejadian, bukan jumlah pekerjaan. Satu work order biasanya
# menghasilkan empat catatan (permintaan, pengeluaran, pengiriman, pemasangan),
# jadi angkanya gampang terbaca jauh lebih besar daripada kenyataannya.
CORRECTIVE_NOTE = (
    "Aktivitas korektif dihitung per CATATAN kejadian, bukan per pekerjaan: "
    "satu work order umumnya menghasilkan beberapa catatan (permintaan, "
    "pengeluaran, pengiriman, pemasangan)."
)

# Kerusakan yang tercatat TIDAK berarti PART berhenti dipakai. PART yang rusak
# masuk bengkel, dan kalau bisa diperbaiki akan dipasang kembali - justru itu
# yang diperkirakan model scrap. Semua PART yang bisa diskor di sini sedang
# terpasang, jadi kerusakan di riwayatnya pasti sudah dilewati.
FAILURE_HISTORY_NOTE = (
    "Kerusakan yang tercatat bukan berarti PART berhenti dipakai: PART yang "
    "rusak masuk bengkel dan dipasang kembali kalau bisa diperbaiki. PART ini "
    "sedang terpasang."
)

RISK, MITIGATING, CONTEXT = "RISK_FACTOR", "MITIGATING", "CONTEXT"

# Kolom snapshot mentah yang dibaca modul ini. Didaftar di sini supaya
# batch_predictor tahu persis apa yang perlu disimpan agar halaman detail bisa
# dilayani tanpa membaca ulang database - dan supaya daftarnya tidak mungkin
# berbeda dari yang benar-benar dipakai di bawah.
SOURCE_COLUMNS = [
    "item_model_code_clean",
    "days_since_installation",
    "total_prior_events",
    "prior_failure_count",
    "prior_failure_365d",
    "prior_corrective_count",
    "prior_corrective_30d",
    "days_since_last_corrective",
    "prior_distinct_places",
    "previous_cycle_lifetime_mean",
    "has_previous_cycle",
    "log_model_failures_90d",
    "model_failure_rate_90d",
    "log_model_fleet_size",
]


def _number(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return default if pd.isna(value) else float(value)


def _factor(code: str, direction: str, label: str, value) -> dict:
    return {"code": code, "direction": direction, "label": label, "value": value}


def risk_factors(row: pd.Series) -> list[dict]:
    """Faktor risiko satu PART dari satu baris snapshot mentah.

    `row` adalah baris hasil feature_builder.current_observations() yang sudah
    dilengkapi attach_history() dan attach_fleet_snapshot() - jadi setiap
    angka di sini benar-benar dipakai model, bukan hitungan terpisah.
    """
    factors: list[dict] = []

    failures_365 = _number(row, "prior_failure_365d")
    failures_total = _number(row, "prior_failure_count")
    if failures_365 > 0:
        factors.append(_factor(
            "RECENT_FAILURE_HISTORY", RISK,
            f"{int(failures_365)} kerusakan tercatat dalam 365 hari terakhir",
            int(failures_365),
        ))
    elif failures_total > 0:
        factors.append(_factor(
            "OLDER_FAILURE_HISTORY", CONTEXT,
            f"{int(failures_total)} kerusakan tercatat sepanjang riwayat PART, "
            "tidak ada dalam 365 hari terakhir",
            int(failures_total),
        ))
    else:
        factors.append(_factor(
            "NO_FAILURE_HISTORY", MITIGATING,
            "Belum pernah tercatat rusak sama sekali", 0,
        ))

    corrective_30 = _number(row, "prior_corrective_30d")
    if corrective_30 > 0:
        factors.append(_factor(
            "RECENT_CORRECTIVE_MAINTENANCE", RISK,
            f"{int(corrective_30)} catatan aktivitas korektif dalam 30 hari terakhir",
            int(corrective_30),
        ))

    days_since_corrective = pd.to_numeric(
        row.get("days_since_last_corrective"), errors="coerce"
    )
    corrective_total = _number(row, "prior_corrective_count")
    if corrective_total > 0 and not pd.isna(days_since_corrective):
        factors.append(_factor(
            "CORRECTIVE_HISTORY", CONTEXT,
            f"{int(corrective_total)} catatan aktivitas korektif sepanjang "
            f"riwayat, terakhir {int(days_since_corrective)} hari lalu",
            int(days_since_corrective),
        ))

    age = _number(row, "days_since_installation")
    oldest_band = config.AGE_BAND_THRESHOLDS[-1]
    factors.append(_factor(
        "INSTALLATION_AGE", RISK if age >= oldest_band else CONTEXT,
        f"Terpasang {int(age)} hari (kelompok umur {_age_band_label(age)})",
        int(age),
    ))

    # Kondisi armada: berapa kerusakan model PART yang sama belakangan.
    fleet_failures = int(round(np.expm1(_number(row, "log_model_failures_90d"))))
    fleet_size = int(round(np.expm1(_number(row, "log_model_fleet_size"))))
    if fleet_failures > 0:
        rate = _number(row, "model_failure_rate_90d")
        factors.append(_factor(
            "FLEET_CONDITION", RISK,
            f"Model PART ini mengalami {fleet_failures} kerusakan dalam "
            f"{config.FLEET_WINDOW_DAYS} hari terakhir dari {fleet_size} unit "
            f"terpasang ({rate:.1%} per unit)",
            round(rate, 4),
        ))

    if bool(row.get("has_previous_cycle", False)):
        lifetime = _number(row, "previous_cycle_lifetime_mean")
        factors.append(_factor(
            "PREVIOUS_CYCLE_LIFETIME", CONTEXT,
            f"Rata-rata umur pemasangan sebelumnya {int(lifetime)} hari",
            int(lifetime),
        ))

    places = _number(row, "prior_distinct_places")
    if places > 1:
        factors.append(_factor(
            "LOCATION_CHANGES", CONTEXT,
            f"Pernah tercatat di {int(places)} lokasi berbeda", int(places),
        ))

    return factors


def caveats(row: pd.Series, support_by_model: dict[str, int]) -> list[str]:
    """Hal yang membuat angka perlu dibaca lebih hati-hati untuk PART ini."""
    notes: list[str] = []
    model_code = row.get("item_model_code_clean")
    support = support_by_model.get(str(model_code), 0) if model_code else 0
    if not model_code:
        notes.append(
            "Model PART tidak diketahui, sehingga fitur identitas PART masuk "
            "kategori UNKNOWN."
        )
    elif support < config.MIN_PART_MODEL_SUPPORT:
        notes.append(
            f"Riwayat model PART '{model_code}' masih sedikit ({support} observasi "
            f"saat training, ambang {config.MIN_PART_MODEL_SUPPORT}), jadi model "
            "menilainya bersama kelompok berdukungan rendah."
        )
    return notes


def _age_band_label(days: float) -> str:
    index = int(np.searchsorted(config.AGE_BAND_THRESHOLDS, days, side="right"))
    return config.AGE_BAND_LABELS[index]
