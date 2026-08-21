"""Bangun dataset event-based: DB -> lifecycle eligible -> landmark -> fitur
dinamis -> split (Tahap 6-9).

    python survival_model/event_based/build_dataset.py

Beda dari `survival_model/build_dataset.py` (model statis, TIDAK diubah oleh
modul ini): satu lifecycle bisa menghasilkan BANYAK baris (landmark), fitur
riwayat/armada dihitung ULANG pada tiap landmark (bukan cuma di installed_on).

Alurnya:

    data_reader.get_cycles()/get_events()/get_failure_episodes()  (REUSE, sama seperti model statis)
        -> lifecycle_builder.cohort_cycles()/assign_lifecycle_outcome()   (REUSE APA ADANYA)
        -> eb_src.landmark_builder.build_landmarks()                      (BARU - lihat modul untuk desain)
        -> feature_builder.attach_history()/attach_fleet()                (REUSE - observation_on=landmark)
        -> install_context.attach_install_context() + previous_cycle       (REUSE - konstan per lifecycle)
        -> eb_src.features.point_in_time_support() + compute_features()   (BARU - lihat modul)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
# TANPA guard "not in sys.path": saat dijalankan sebagai skrip, Python
# OTOMATIS menyisipkan direktori skrip ini sendiri (EVENT_BASED_DIR) ke
# sys.path[0] SEBELUM baris ini jalan - guard "not in sys.path" jadi True
# lalu insert di atas DILEWATI, dan EVENT_BASED_DIR tetap di posisinya
# semula (index 0) SAMPAI insert ROOT_DIR/SURVIVAL_DIR di atas mendorongnya
# ke urutan KE-3 (di BELAKANG SURVIVAL_DIR) - `import build_dataset` lalu
# salah resolve ke survival_model/build_dataset.py (statis, nama modul
# sama). Insert TANPA syarat di sini memaksa EVENT_BASED_DIR balik ke index
# 0, apa pun yang terjadi sebelumnya.
sys.path.insert(0, str(EVENT_BASED_DIR))

import os

import joblib
import pandas as pd

import data_reader
import feature_builder

from src import install_context, lifecycle_builder, previous_cycle

from eb_src import features, landmark_builder

REPORTS_DIR = EVENT_BASED_DIR / "reports"
_DEV_CACHE_PATH = EVENT_BASED_DIR / "artifacts" / "_experiment_cache" / "build_dataset_build.joblib"


def build() -> dict:
    """Baca database dan susun dataset event-based siap dipakai train.py.

    Set env var SURVIVAL_BUILD_CACHE=1 untuk memakai cache lokal (dev/debug -
    sama seperti model statis, TIDAK aktif default)."""
    if os.environ.get("SURVIVAL_BUILD_CACHE") and _DEV_CACHE_PATH.exists():
        print("      [dev cache] memuat event_based/build_dataset.build() dari cache lokal...")
        return joblib.load(_DEV_CACHE_PATH)

    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()
    data_end = pd.Timestamp(cycles["dataset_max_event_on"].max())

    cohort = lifecycle_builder.cohort_cycles(cycles)
    outcome = lifecycle_builder.assign_lifecycle_outcome(cohort, data_end)

    print("      Membangun landmark (INSTALL + event organik + anchor jarang)...")
    landmarks = landmark_builder.build_landmarks(outcome, events)

    # --- Konteks KONSTAN per lifecycle (install context + terminal) - ditempel
    # LEBIH DULU (sebelum dukungan/fitur dinamis) karena point_in_time_support()
    # untuk item_type_at_install/terminal_type_context butuh kolom ini sudah
    # ada di landmarks. terminal_raw = query kanonikal BARU (data_reader.py,
    # BUKAN schema analytics - lihat eb_src/features.py & terminal_context.py
    # docstring untuk penjelasan lengkap + verifikasi angka).
    landmarks = install_context.attach_install_context(landmarks, events)
    terminal_raw = data_reader.get_terminal_context()
    landmarks = features.attach_terminal_extra(landmarks, terminal_raw)

    # --- Dukungan point-in-time: dari cohort PENUH (SATU baris per lifecycle,
    # BUKAN dari landmarks yang sudah diperbanyak) - lihat eb_src/features.py
    # docstring bagian "TIDAK reuse" untuk alasan kenapa ini WAJIB terpisah.
    cohort_with_type = install_context.attach_install_context(cohort, events)
    cohort_with_terminal = features.attach_terminal_extra(cohort, terminal_raw)
    support = features.point_in_time_support(landmarks, cohort, "item_model_code_clean")
    item_type_support = features.point_in_time_support(landmarks, cohort_with_type, "item_type_at_install")
    terminal_support = features.point_in_time_support(landmarks, cohort_with_terminal, "terminal_type_context")

    # --- Fitur dinamis: riwayat/armada dihitung ULANG pada observation_on
    # tiap landmark - feature_builder.attach_history/attach_fleet SUDAH
    # generik terhadap kolom observation_on, tidak ada logic baru di sini.
    landmarks["days_since_installation"] = landmarks["landmark_age_days"]
    landmarks = feature_builder.attach_history(landmarks, events)
    landmarks = feature_builder.attach_fleet(landmarks, cycles, episodes)

    # --- Previous-cycle confirmed-failure (konstan per lifecycle, bicara
    # tentang siklus SEBELUMNYA) - merge apa adanya.
    pc = previous_cycle.audit_previous_cycle_features(cycles)
    # transform_for_model() butuh KEDUA kolom audit (confirmed-failure-mean
    # DAN last-confirmed) walau hanya confirmed-failure-mean yang dipakai di
    # fitur FINAL (sama seperti model statis src/features.attach_final_context) -
    # merge tanpa last_confirmed_failure_lifetime akan KeyError di dalam
    # transform_for_model().
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

    # --- Fitur dinamis TAMBAHAN (hasil experiments.py ablation): degradation
    # trend + cumulative physical usage + jendela corrective 60/90 hari
    # (konfigurasi G_combined_without_device, VAL t0-only 0,7849 -> 0,7954).
    landmarks = features.attach_dynamic_extra(landmarks, cycles, events)

    feature_frame = features.compute_features(landmarks, support, item_type_support, terminal_support)

    dataset = landmarks[[
        "installation_cycle_id", "item_identifier_clean", "installed_on", "observation_on",
        "landmark_age_days", "landmark_source", "item_model_code_clean", "failure_onset_on",
        "cycle_end_on", "cycle_end_reason", "split", "cutoff_on", "duration_days", "event_observed",
    ]].reset_index(drop=True)

    result = {
        "dataset": dataset,
        "features": feature_frame,
        "support_totals": _support_totals(cohort, "item_model_code_clean"),
        "item_type_support_totals": _support_totals(cohort_with_type, "item_type_at_install"),
        "terminal_support_totals": _support_totals(cohort_with_terminal, "terminal_type_context"),
        "data_end": data_end,
        "events": events,
        "cycles": cycles,
        "episodes": episodes,
        "outcome": outcome,
        "landmarks": landmarks,
    }
    if os.environ.get("SURVIVAL_BUILD_CACHE"):
        _DEV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, _DEV_CACHE_PATH)
    return result


def _support_totals(baseline: pd.DataFrame, column: str) -> dict[str, int]:
    """Dukungan AKHIR (dibekukan ke metadata, dipakai predict.py) - jumlah
    lifecycle per kategori pada `data_end`, populasi SAMA dengan yang dipakai
    `features.point_in_time_support()` (cohort penuh, satu baris/lifecycle)."""
    totals = baseline.groupby(column).size()
    return {str(key): int(count) for key, count in totals.items()}


def main() -> int:
    built = build()
    dataset = built["dataset"]
    print(f"      Total landmark rows: {len(dataset):,}")
    for split_name in ("TRAIN", "VALIDATION", "TEST"):
        mask = dataset["split"] == split_name
        print(
            f"      {split_name:12s} rows={int(mask.sum()):,}  "
            f"events={int(dataset.loc[mask, 'event_observed'].sum()):,}  "
            f"lifecycles={dataset.loc[mask, 'installation_cycle_id'].nunique():,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
