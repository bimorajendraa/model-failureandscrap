"""Bangun dataset survival: DB -> lifecycle eligible -> fitur baseline -> split.

    python survival_model/build_dataset.py

Alurnya:

    data_reader.get_cycles()  -> cohort_cycles()          siklus dgn identitas model konsisten
                              -> assign_lifecycle_outcome() split + cutoff + duration/event/eligible
                              -> eligible_lifecycles()      subset yang labelnya bisa dipastikan
                              -> attach_survival_features()  riwayat + armada point-in-time
                              -> compute_features()          19 fitur baseline instalasi

Juga menjalankan pemeriksaan kualitas data (README bagian "Validasi hasil")
dan menulis `reports/data_validation.md` - dilaporkan, bukan didiamkan.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import numpy as np
import pandas as pd

import data_reader
import feature_builder

from src import features, lifecycle_builder, utils

REPORTS_DIR = SURVIVAL_DIR / "reports"


def build() -> dict:
    """Baca database dan susun dataset survival siap dipakai train.py/evaluate.py."""
    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()
    data_end = pd.Timestamp(cycles["dataset_max_event_on"].max())

    cohort = lifecycle_builder.cohort_cycles(cycles)
    outcome = lifecycle_builder.assign_lifecycle_outcome(cohort, data_end)

    # Dukungan historis tipe PART dihitung dari SELURUH cohort (bukan hanya
    # baris eligible) - sama seperti model classification
    # (feature_builder.training_observations): kalau dihitung hanya dari
    # baris eligible, PART model yang banyak siklus REINSTALL/RECON-ambigu
    # akan tampak lebih jarang daripada sebenarnya.
    baseline_all = features.build_baseline_observations(outcome)
    support_all = feature_builder.cumulative_support(baseline_all)
    support_totals = feature_builder.support_totals(baseline_all)
    outcome = outcome.assign(_support=support_all.to_numpy())

    lifecycles = lifecycle_builder.eligible_lifecycles(outcome)

    observations = features.build_baseline_observations(lifecycles)
    observations = features.attach_survival_features(observations, events, cycles, episodes)
    feature_frame = features.compute_features(observations, observations["_support"])

    dataset = observations[[
        "installation_cycle_id", "item_identifier_clean", "installed_on",
        "item_model_code_clean", "failure_onset_on", "cycle_end_on",
        "cycle_end_reason", "split", "cutoff_on", "duration_days", "event_observed",
    ]].reset_index(drop=True)

    return {
        "dataset": dataset,
        "features": feature_frame,
        "support_totals": support_totals,
        "data_end": data_end,
        "events": events,
        "cycles": cycles,
        "episodes": episodes,
        "cohort_outcome": outcome,  # dipakai validate() untuk melaporkan yang di-exclude
    }


# ---------------------------------------------------------------------------
# Validasi (README bagian "Validasi hasil") - dicetak + ditulis ke reports/,
# TIDAK diam-diam membuang data aneh.
# ---------------------------------------------------------------------------


def validate(built: dict) -> str:
    dataset = built["dataset"]
    outcome = built["cohort_outcome"]
    lines: list[str] = ["# Validasi dataset survival", ""]

    def add(title: str, body: str) -> None:
        lines.append(f"## {title}")
        lines.append(body)
        lines.append("")

    total_cohort = len(outcome)
    total_eligible = len(dataset)
    add(
        "Jumlah lifecycle",
        f"Cohort (is_initial_model_cohort, durasi positif): {total_cohort:,}\n\n"
        f"Eligible untuk survival (lolos aturan censoring per-split): "
        f"{total_eligible:,} ({total_eligible / total_cohort:.1%} dari cohort)",
    )

    excluded = outcome.loc[~outcome["eligible"]]
    by_reason = (
        excluded.groupby(["split", "cycle_end_reason"], dropna=False).size().rename("count").reset_index()
    )
    add(
        "Lifecycle yang di-exclude, per split & alasan siklus berakhir",
        by_reason.to_string(index=False) if len(by_reason) else "(tidak ada yang di-exclude)",
    )

    counts = dataset.groupby(["split", "event_observed"]).size().unstack(fill_value=0)
    counts.columns = ["censored (0)", "event (1)"] if list(counts.columns) == [0, 1] else counts.columns
    add("Event vs censored per split", counts.to_string())

    duration = dataset["duration_days"]
    add(
        "Distribusi duration_days",
        (
            f"min={duration.min():.1f}  p25={duration.quantile(.25):.1f}  "
            f"median={duration.median():.1f}  p75={duration.quantile(.75):.1f}  "
            f"p99={duration.quantile(.99):.1f}  max={duration.max():.1f}"
        ),
    )

    n_nonpositive = int((duration <= 0).sum())
    n_dupe = int(dataset["installation_cycle_id"].duplicated().sum())
    n_failure_before_install = int(
        (dataset["event_observed"].eq(1) & (dataset["failure_onset_on"] < dataset["installed_on"])).sum()
    )
    n_future_install = int((dataset["installed_on"] > pd.Timestamp.now()).sum())
    add(
        "Cek integritas (semua harus 0)",
        (
            f"duration_days <= 0: {n_nonpositive}\n\n"
            f"installation_cycle_id duplikat: {n_dupe}\n\n"
            f"failure_onset_on < installed_on (pada lifecycle event=1): {n_failure_before_install}\n\n"
            f"installed_on di masa depan (> sekarang): {n_future_install}"
        ),
    )

    train_categories = set(dataset.loc[dataset["split"] == "TRAIN", "item_model_code_clean"].dropna())
    for split_name in ("VALIDATION", "TEST"):
        split_categories = set(dataset.loc[dataset["split"] == split_name, "item_model_code_clean"].dropna())
        unseen = split_categories - train_categories
        add(
            f"Tipe PART yang cuma muncul di {split_name} (tidak pernah di TRAIN)",
            f"{len(unseen)} tipe dari {len(split_categories)} tipe di {split_name} "
            "(catatan: part_model_category sudah mengelompokkan tipe bersupport rendah "
            "jadi satu kategori bersama - lihat config.MIN_PART_MODEL_SUPPORT; "
            "OneHotEncoder di train.py juga diberi handle_unknown='ignore' sebagai pengaman kedua)",
        )

    item_split_counts = dataset.groupby("item_identifier_clean")["split"].nunique()
    cross_split_items = int((item_split_counts > 1).sum())
    add(
        "PART dengan lifecycle di lebih dari satu split",
        f"{cross_split_items:,} item ({cross_split_items / dataset['item_identifier_clean'].nunique():.1%} "
        "dari item unik) punya >1 lifecycle yang jatuh di split berbeda (mis. cycle 1 di TRAIN, "
        "cycle 3 di TEST). Bukan leakage temporal (urutan waktu tetap terjaga), tapi potensi model "
        "'mengenali' identitas item lintas split lewat fitur previous_cycle_lifetime_mean/"
        "has_previous_cycle. Didokumentasikan sebagai keterbatasan, tidak diperbaiki dengan grouped "
        "split (lihat README).",
    )

    rate_by_split = dataset.groupby("split")["event_observed"].mean()
    add(
        "Base rate event (failure) per split - untuk melihat pergeseran",
        rate_by_split.to_string(),
    )

    report = "\n".join(lines)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_validation.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    print("[1/2] Membaca database dan menyusun lifecycle survival...")
    built = build()
    dataset = built["dataset"]
    print(
        f"      {len(built['cohort_outcome']):,} lifecycle cohort -> {len(dataset):,} eligible "
        f"({int(dataset['event_observed'].sum()):,} event, "
        f"{int((dataset['event_observed'] == 0).sum()):,} censored)"
    )
    print("[2/2] Menjalankan validasi data...")
    report = validate(built)
    print(report)
    print(f"\n[OK] Laporan tersimpan di {REPORTS_DIR / 'data_validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
