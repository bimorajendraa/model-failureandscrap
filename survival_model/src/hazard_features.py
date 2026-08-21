"""Fitur hazard tambahan (sesi peningkatan C-index, Fase 2 - lihat
reports/hazard_ablation.md untuk hasil audit).

Ablation lama (README poin 5, `reports/feature_ablation.md`) sudah
membuktikan konteks statis (part model/client/lokasi TANPA riwayat) jauh di
bawah fitur riwayat kejadian (VAL C-index 0,65 vs 0,81) - jadi arah di sini
BUKAN menambah konteks statis baru, melainkan memperdalam sinyal riwayat
yang sudah terbukti kuat: bukan cuma "berapa kali corrective/failure
terjadi" (fitur lama, `feature_builder.attach_history`), tapi "part model/
tipe/client yang sama biasanya bertahan berapa lama sebelum gagal" - prior
survival empiris, target yang PERSIS sama dengan yang diprediksi model ini.

Point-in-time safe dengan mekanisme yang SAMA dengan
`feature_builder.attach_fleet` (dipakai fitur `log_model_failures_90d` dkk
yang SUDAH ada di fitur final): hanya lifecycle LAIN yang cycle_end_on-nya
STRICTLY SEBELUM installed_on baris ini yang ikut dihitung - TIDAK ada
query database baru (`cycles` sudah dibaca `data_reader.get_cycles()` lewat
build_dataset.build()), TIDAK ada mapping kanonikal baru.

Dihitung dari populasi `is_initial_model_cohort` PENUH (sama seperti
attach_fleet, BUKAN dibatasi ke lifecycle eligible survival) - supaya
statistik "part model ini biasanya bertahan berapa lama" tidak bias oleh
aturan censoring per-split (lifecycle yang di-exclude survival karena
statusnya ambigu di cutoff split-nya sendiri TETAP sah dipakai sebagai
riwayat historis murni cross-sectional di sini, sama seperti fitur armada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_prior_survival(
    observations: pd.DataFrame, cycles: pd.DataFrame, group_column: str, prefix: str
) -> pd.DataFrame:
    """Untuk tiap baris `observations` (butuh kolom `installed_on` dan
    `group_column`) - statistik lifecycle LAIN pada `group_column` yang
    SAMA, yang sudah BERAKHIR (`cycle_end_on < installed_on` baris ini,
    ketat) sebelum baris ini dipasang:

    - `{prefix}_prior_ended_count` / `log_{prefix}_prior_ended_count`:
      berapa lifecycle grup ini yang sudah berakhir (sebelum baris ini)
    - `{prefix}_prior_failure_share`: proporsi yang berakhir FAILURE
      (0 kalau belum ada yang berakhir sama sekali)
    - `{prefix}_prior_failure_lifetime_median` /
      `log_{prefix}_prior_failure_lifetime_median` /
      `has_{prefix}_prior_failure_lifetime`: median durasi di antara yang
      berakhir FAILURE (NaN/0/False kalau belum ada FAILURE tercatat pada
      grup ini sebelum baris ini - genuinely tidak diketahui, BUKAN
      diasumsikan 0 - lihat `has_*`)

    `group_column` boleh apa pun yang ada di `cycles` DAN `observations`
    dengan nama sama (`item_model_code_clean`, `item_type_at_install`,
    `installed_client_clean`, dst.) - nilai kosong dikelompokkan sebagai
    "UNKNOWN" sendiri (tidak dicampur ke grup manapun).
    """
    # reset_index(drop=True): pola yang SAMA dengan feature_builder.
    # attach_history/attach_fleet - groupby(...).indices di bawah
    # mengembalikan LABEL index, yang hanya sama dengan POSISI array numpy
    # (dipakai indexing at[...]/ended_count[...]) kalau index-nya RangeIndex
    # bersih. observations dari features.build_baseline_observations() sudah
    # begini, reset ulang di sini hanya jaring pengaman.
    observations = observations.reset_index(drop=True)
    cohort = cycles.loc[cycles["is_initial_model_cohort"].fillna(False)].copy()
    cohort["_duration"] = (cohort["cycle_end_on"] - cohort["installed_on"]) / np.timedelta64(1, "D")
    cohort["_is_failure"] = cohort["cycle_end_reason"].eq("FAILURE")
    cohort["_group_key"] = cohort[group_column].fillna("UNKNOWN").astype(str)

    obs_keys = observations[group_column].fillna("UNKNOWN").astype(str)
    at = observations["installed_on"].to_numpy("datetime64[ns]")

    n = len(observations)
    ended_count = np.zeros(n, dtype="int64")
    failure_count = np.zeros(n, dtype="int64")
    failure_median = np.full(n, np.nan)

    grouped_cohort = cohort.groupby("_group_key", sort=False)
    obs_rows_by_key = obs_keys.groupby(obs_keys, sort=False).indices

    for key, obs_positions in obs_rows_by_key.items():
        if key not in grouped_cohort.groups:
            continue
        group_cycles = grouped_cohort.get_group(key).sort_values("cycle_end_on", kind="stable")
        ended_sorted = group_cycles["cycle_end_on"].to_numpy("datetime64[ns]")
        is_failure_sorted = group_cycles["_is_failure"].to_numpy()
        duration_sorted = group_cycles["_duration"].to_numpy()

        cum_failure = np.cumsum(is_failure_sorted)
        # Median durasi FAILURE "sampai posisi ini" (expanding, per posisi
        # cycle yang SUDAH berakhir dalam grup - divektorkan lewat pandas
        # .expanding().median(), BUKAN loop python per baris observasi,
        # supaya tetap cepat pada grup besar seperti MODULE READER ~5rb baris).
        failure_only = pd.Series(np.where(is_failure_sorted, duration_sorted, np.nan))
        expanding_median = failure_only.expanding().median().to_numpy()

        positions = obs_positions.to_numpy() if hasattr(obs_positions, "to_numpy") else np.asarray(obs_positions)
        query_times = at[positions]
        # side="left": jumlah cycle_end_on STRICT < installed_on baris ini -
        # mekanisme sama dengan feature_builder.attach_fleet._count_before.
        pos = np.searchsorted(ended_sorted, query_times, side="left")

        ended_count[positions] = pos
        has_prior = pos > 0
        idx_prior = np.clip(pos - 1, 0, None)
        failure_count[positions] = np.where(has_prior, cum_failure[idx_prior], 0)
        failure_median[positions] = np.where(has_prior, expanding_median[idx_prior], np.nan)

    out = pd.DataFrame(index=observations.index)
    out[f"{prefix}_prior_ended_count"] = ended_count
    out[f"log_{prefix}_prior_ended_count"] = np.log1p(ended_count)
    out[f"{prefix}_prior_failure_share"] = np.divide(
        failure_count, ended_count, out=np.zeros(n, dtype=float), where=ended_count > 0
    )
    out[f"has_{prefix}_prior_failure_lifetime"] = ~np.isnan(failure_median)
    out[f"log_{prefix}_prior_failure_lifetime_median"] = np.log1p(np.nan_to_num(failure_median, nan=0.0))
    return out
