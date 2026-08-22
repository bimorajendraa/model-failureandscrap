"""Skor model survival event-based pada kondisi SEKARANG (observation_on
tiap baris), bukan dibekukan di installed_on - scorer promosi permanen untuk
model survival, hasil Fase A1 (lihat survival_model/event_based/reports/gate_decision.md).

MASALAH yang diselesaikan: mengevaluasi model survival dengan fitur dibekukan
di installed_on (t0-only) menghukum model ini justru pada sumbu yang jadi
alasannya dibangun - fitur dinamis yang di-refresh seiring waktu.
`predict.survival` menskor pada KONDISI SEKARANG, bukan kondisi instalasi -
evaluasi yang representatif harus melakukan hal yang sama. Untuk SETIAP baris
populasi yang mau dinilai, fitur event-based dibangun PERSIS pada
`observation_on` baris itu sendiri - baris itu diperlakukan sebagai satu
landmark tunggal, mekanisme SAMA PERSIS dengan satu landmark di
`features/survival/landmarks.py`, hanya titik waktunya beda.

TIGA JEBAKAN yang masing-masing bisa memalsukan hasil:

1. risk_30d = 1 - S(30) LANGSUNG, BUKAN survival.curves.conditional_risk().
   Kurva event-based sudah bermula di t=0=observation_on (predict/survival.py)
   - conditional_risk (rumus 1-S(age+30)/S(age)) untuk model yang kurvanya
   dari t=0=installed_on, salah dua kali kalau dipakai di sini.
2. Support totals (part_model/item_type/terminal) WAJIB dari dict BEKU hasil
   training (metadata.json) - dipetakan persis seperti predict/survival.py,
   BUKAN dihitung ulang dari frame yang dinilai. Menghitung ulang = leakage.
3. Memori: banyak baris x predict_survival_function pada RSF besar bisa OOM -
   di-chunk, kurva dibuang tiap chunk.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from partrisk.features import failure as feature_builder
from partrisk.features.survival import builder as eb_features
from partrisk.features.survival import install_context, previous_cycle
from partrisk.survival import curves as survival_curves

CHUNK_SIZE = 2000
HORIZON_DAYS = 30.0


def build_landmark_features_at_observation(
    rows: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, episodes: pd.DataFrame,
    terminal_raw: pd.DataFrame, metadata: dict,
) -> pd.DataFrame:
    """Fitur event-based PADA observation_on tiap baris - JEBAKAN #2 (support
    beku) ditegakkan di sini. `rows` butuh kolom `days_since_installation`
    (umur landmark) dan seluruh kolom identitas yang dipakai
    `install_context`/`feature_builder` (installation_cycle_id,
    item_identifier_clean, installed_on, item_model_code_clean, dst)."""
    landmarks = rows.reset_index(drop=True).copy()
    landmarks["landmark_age_days"] = landmarks["days_since_installation"]

    landmarks = install_context.attach_install_context(landmarks, events)
    landmarks = eb_features.attach_terminal_extra(landmarks, terminal_raw)

    landmarks["days_since_installation"] = landmarks["landmark_age_days"]
    landmarks = feature_builder.attach_history(landmarks, events)
    landmarks = feature_builder.attach_fleet(landmarks, cycles, episodes)

    pc = previous_cycle.audit_previous_cycle_features(cycles)
    landmarks = landmarks.merge(
        pc[[
            "installation_cycle_id",
            "previous_cycle_confirmed_failure_lifetime_mean",
            "last_confirmed_failure_lifetime",
        ]],
        on="installation_cycle_id", how="left",
    )
    transform = previous_cycle.transform_for_model(landmarks)[
        ["log_previous_cycle_confirmed_failure_lifetime_mean", "has_previous_cycle_confirmed_failure_lifetime_mean"]
    ]
    landmarks = pd.concat([landmarks, transform], axis=1)

    landmarks = eb_features.attach_dynamic_extra(landmarks, cycles, events)
    # compute_features() di bawah memanggil feature_builder.build_features()
    # sebagai utilitas bersama (reuse kolom classification) - fungsi itu
    # sekarang juga butuh log_previous_cycle_count (config.DEGRADATION_FEATURES).
    # attach_dynamic_extra() di atas sudah punya previous_cycle_count RAW
    # (dipakai survival sendiri, nama BEDA sengaja - lihat
    # DYNAMIC_EXTRA_NUMERIC_COLUMNS) - tinggal di-log1p, bukan dihitung ulang.
    landmarks["log_previous_cycle_count"] = np.log1p(landmarks["previous_cycle_count"].astype(float))

    # JEBAKAN #2: support DIBEKUKAN dari metadata.json (hasil training),
    # dipetakan persis seperti predict/survival.py - TIDAK dihitung ulang
    # dari populasi yang dinilai.
    support_totals = {k: int(v) for k, v in metadata["support_totals"].items()}
    item_type_support_totals = {k: int(v) for k, v in metadata["item_type_support_totals"].items()}
    terminal_support_totals = {k: int(v) for k, v in metadata["terminal_support_totals"].items()}

    support = landmarks["item_model_code_clean"].map(support_totals).fillna(0).astype("int64")
    item_type_support = landmarks["item_type_at_install"].map(item_type_support_totals).fillna(0).astype("int64")
    terminal_support = landmarks["terminal_type_context"].map(terminal_support_totals).fillna(0).astype("int64")

    feature_frame = eb_features.compute_features(landmarks, support, item_type_support, terminal_support)
    return feature_frame.reset_index(drop=True)


def score_risk_30d_chunked(model, feature_frame: pd.DataFrame, encoder) -> np.ndarray:
    """JEBAKAN #3: chunk supaya predict_survival_function tidak OOM. JEBAKAN
    #1: risk = 1 - S(30) LANGSUNG (kurva sudah dari t=0=observation_on)."""
    n = len(feature_frame)
    risk = np.empty(n, dtype=float)
    n_chunks = math.ceil(n / CHUNK_SIZE)
    for i in range(n_chunks):
        lo, hi = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, n)
        chunk = feature_frame.iloc[lo:hi]
        x_chunk = eb_features.encode(chunk, encoder)
        times_grid, curve_values = survival_curves.survival_curve_arrays(model, x_chunk)
        s30 = survival_curves.step_eval_matrix(times_grid, curve_values, [HORIZON_DAYS])[:, 0]
        risk[lo:hi] = 1.0 - s30
        del curve_values
        print(f"      chunk {i+1}/{n_chunks} ({hi:,}/{n:,} baris)...")
    return risk
