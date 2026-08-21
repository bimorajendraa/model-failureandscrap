"""Pembentukan observasi (baris dasar sebelum fitur ditempel) - diekstrak
dari `feature_builder.py` (Fase B2 restrukturisasi), logic TIDAK diubah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from partrisk import config

_DAY = np.timedelta64(1, "D")


def training_observations(cycles: pd.DataFrame) -> pd.DataFrame:
    """Snapshot training pada grid tetap 30 hari sejak tanggal pemasangan.

    Grid dibuat untuk SELURUH siklus dalam cohort (bukan hanya yang layak
    dilatih) karena dukungan historis tipe PART dihitung dari seluruh
    observasi; penyaringan kelayakan dilakukan setelahnya.
    """
    cohort = cycles.loc[
        cycles["is_initial_model_cohort"].fillna(False)
        & (cycles["installed_on"] < cycles["cycle_end_on"])
    ].reset_index(drop=True)

    installed = cohort["installed_on"].to_numpy("datetime64[ns]")
    ends = cohort["cycle_end_on"].to_numpy("datetime64[ns]")
    step = np.timedelta64(config.OBSERVATION_STEP_DAYS, "D")

    # Observasi terakhir harus benar-benar SEBELUM siklus berakhir.
    span = ends - installed - np.timedelta64(1, "us")
    n_steps = (span // step).astype("int64") + 1

    row = np.repeat(np.arange(len(cohort)), n_steps)
    offset = np.arange(n_steps.sum()) - np.repeat(
        np.cumsum(n_steps) - n_steps, n_steps
    )
    observations = cohort.iloc[row].reset_index(drop=True)
    observations["observation_on"] = installed[row] + offset * step

    failure = observations["failure_onset_on"].to_numpy("datetime64[ns]")
    observed = observations["observation_on"].to_numpy("datetime64[ns]")
    horizon = np.timedelta64(config.TARGET_HORIZON_DAYS, "D")
    observations["target_failure"] = (failure > observed) & (
        failure <= observed + horizon
    )

    # Sebuah baris hanya dipakai kalau hasilnya benar-benar bisa dipastikan:
    # positif kalau failure terjadi dalam horizon, negatif kalau 30 hari ke
    # depan sudah sepenuhnya terekam DAN siklusnya memang layak jadi negatif.
    observations["is_eligible"] = observations["target_failure"] | (
        observations["is_recon_verified_negative_eligible"].fillna(False)
        & (
            observations["observation_on"]
            <= observations["last_confirmable_observation_on"]
        )
    )
    return _finalize_observations(observations)


def current_observations(cycles: pd.DataFrame) -> pd.DataFrame:
    """Satu snapshot per PART yang saat ini masih terpasang.

    Diambil pada kejadian terbaru yang tercatat, BUKAN pada grid 30 hari
    seperti dataset training - kalau memakai grid, skor sebuah PART bisa
    tertinggal sampai ~29 hari dari kondisi terakhir yang sudah diketahui.
    """
    active = cycles.loc[
        cycles["is_initial_model_cohort"].fillna(False)
        & cycles["cycle_end_reason"].eq("RIGHT_CENSORED_AT_DATA_END")
    ].copy()
    active["observation_on"] = active["dataset_max_event_on"]
    return _finalize_observations(active)


def _finalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.reset_index(drop=True)
    observations["days_since_installation"] = (
        observations["observation_on"].to_numpy("datetime64[ns]")
        - observations["installed_on"].to_numpy("datetime64[ns]")
    ) / _DAY
    return observations
