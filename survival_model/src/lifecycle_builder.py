"""Bangun unit observasi survival: satu baris per lifecycle (installation
cycle), dengan duration/event yang dihitung terhadap batas administrative
censoring MASING-MASING split (bukan satu cutoff global).

Sumber datanya `data_reader.get_cycles()` - SUDAH satu baris per siklus
pemasangan lengkap dengan cara siklus itu berakhir (FAILURE /
RIGHT_CENSORED_AT_DATA_END / REINSTALL_WITHOUT_RECORDED_FAILURE). Modul ini
TIDAK membangun ulang lifecycle dari event mentah - hanya menurunkan label
survival dari tabel yang sudah ada.

Kenapa cutoff per-split (bukan satu dataset_max_event_on untuk semua baris):
kalau lifecycle TRAIN yang dipasang lama tetap disensor di tanggal data
TERBARU, labelnya diam-diam membawa informasi tentang apa yang terjadi
sepanjang periode VALIDATION/TEST - versi survival dari alasan model
classification butuh embargo. Solusinya di sini: setiap baris disensor pada
batas SPLIT-nya sendiri (validation_start untuk TRAIN, test_start untuk
VALIDATION, data_end untuk TEST), dihitung ulang dari fakta yang sudah ada
di `cycle_end_on`/`failure_onset_on` - bukan exclude berbasis embargo seperti
classification, karena durasi survival tidak punya window tetap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils

# Kolom yang wajib ada di `cycles` (dari data_reader.get_cycles()).
_REQUIRED_COLUMNS = [
    "installation_cycle_id", "item_identifier_clean", "installed_on",
    "item_model_code_clean", "installed_client_clean", "failure_onset_on",
    "cycle_end_on", "cycle_end_reason", "is_recon_verified_negative_eligible",
    "is_initial_model_cohort", "previous_cycle_lifetime_mean", "has_previous_cycle",
]


def cohort_cycles(cycles: pd.DataFrame) -> pd.DataFrame:
    """Siklus dengan identitas model PART yang konsisten dan durasi positif -
    filter yang SAMA dengan cohort model classification
    (`feature_builder.training_observations`), supaya `part_model_category`
    berarti hal yang sama di kedua model."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in cycles.columns]
    if missing:
        raise ValueError(f"Kolom hilang dari data_reader.get_cycles(): {missing}")
    return cycles.loc[
        cycles["is_initial_model_cohort"].fillna(False)
        & (cycles["installed_on"] < cycles["cycle_end_on"])
    ].reset_index(drop=True)


def assign_lifecycle_outcome(cohort: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """Tambahkan split, cutoff, duration_days, event_observed, eligible.

    Aturan (berlaku sama untuk ketiga split - untuk TEST, cutoff=data_end
    membuatnya otomatis identik dengan cutoff global lama):

        kalau failure_onset_on ADA dan <= cutoff:
            event=1, duration = failure_onset_on - installed_on   (fakta historis)
        elif cycle_end_on > cutoff:
            event=0, duration = cutoff - installed_on             (bukti langsung:
                                masih berjalan tepat di cutoff, apa pun nasib
                                akhirnya - REINSTALL atau RECON ambigu yang
                                terjadi SETELAH cutoff jadi tidak relevan)
        else:  # cycle sudah berakhir pada/sebelum cutoff, tanpa failure -
               # hanya mungkin saat cutoff=data_end (TEST)
            kalau RIGHT_CENSORED_AT_DATA_END dan is_recon_verified_negative_eligible:
                event=0, duration = cycle_end_on - installed_on
            selain itu: EXCLUDE - status pada cutoff tidak bisa dipastikan
    """
    outcome = cohort.copy()
    outcome["split"] = utils.assign_lifecycle_split(outcome["installed_on"], data_end)
    validation_start, test_start = utils.lifecycle_split_bounds(data_end)
    cutoff_by_split = {
        utils.TRAIN: validation_start,
        utils.VALIDATION: test_start,
        utils.TEST: pd.Timestamp(data_end),
    }
    outcome["cutoff_on"] = outcome["split"].map(cutoff_by_split)

    installed = outcome["installed_on"].to_numpy("datetime64[ns]")
    cutoff = pd.to_datetime(outcome["cutoff_on"]).to_numpy("datetime64[ns]")
    failure = outcome["failure_onset_on"].to_numpy("datetime64[ns]")
    cycle_end = outcome["cycle_end_on"].to_numpy("datetime64[ns]")
    day = np.timedelta64(1, "D")

    has_cutoff = ~pd.isna(outcome["cutoff_on"]).to_numpy()
    has_failure_before_cutoff = ~pd.isna(outcome["failure_onset_on"]).to_numpy() & has_cutoff & (failure <= cutoff)
    still_running_at_cutoff = has_cutoff & (cycle_end > cutoff)
    verified_censored_at_end = (
        has_cutoff
        & outcome["cycle_end_reason"].eq("RIGHT_CENSORED_AT_DATA_END").to_numpy()
        & outcome["is_recon_verified_negative_eligible"].fillna(False).to_numpy()
        & (cycle_end <= cutoff)
    )
    positive_span = has_cutoff & (installed < cutoff)

    eligible = positive_span & (has_failure_before_cutoff | still_running_at_cutoff | verified_censored_at_end)

    duration_days = np.where(has_failure_before_cutoff, (failure - installed) / day, (cutoff - installed) / day)
    # Dibulatkan ke hari bulat: presisi jam/menit dari timestamp mentah tidak
    # berarti apa pun secara bisnis untuk "berapa lama PART bertahan", dan
    # RandomSurvivalForest menyimpan satu titik kurva survival PER waktu unik
    # DI SETIAP leaf node - tanpa pembulatan ini grid waktu unik membengkak
    # sampai ribuan titik (presisi sub-hari dari ratusan ribu timestamp
    # berbeda), yang terbukti membuat artifact model >4 GiB. np.maximum
    # menjaga durasi tetap positif untuk lifecycle yang sangat pendek.
    outcome["duration_days"] = np.maximum(np.round(duration_days), 1.0)
    outcome["event_observed"] = has_failure_before_cutoff.astype(int)
    outcome["eligible"] = eligible
    return outcome


def eligible_lifecycles(outcome: pd.DataFrame) -> pd.DataFrame:
    """Subset baris yang lolos aturan di assign_lifecycle_outcome()."""
    return outcome.loc[outcome["eligible"]].reset_index(drop=True)
