"""Label dan fitur untuk model risiko scrap.

Menjawab: saat sebuah PART rusak, apakah kerusakan itu berakhir dibuang
(UNREPAIRABLE/BROKEN) atau PART kembali dipakai?

SATU-SATUNYA tempat label dan fitur scrap dihitung, dipakai bersama oleh
train_scrap.py dan predict_scrap.py, supaya fitur saat model belajar dan fitur
saat production dijamin sama.

CARA LABEL DITENTUKAN - dua sumber bukti, bukan hanya vonis bengkel:

- DIBUANG   : vonis bengkel UNREPAIRABLE atau BROKEN.
- DIPERBAIKI: vonis bengkel REPAIRED, ATAU PART yang sama terbukti dipasang
  kembali setelah kerusakan itu. Pemasangan ulang adalah bukti langsung PART
  kembali dipakai.
- TIDAK BISA DILABELI: tidak keduanya. Bisa jadi dibuang tanpa dicatat, bisa
  jadi masih di bengkel - tidak ada cara membedakannya, jadi tidak dipakai.

Memakai vonis bengkel saja akan membuang ratusan episode yang sebenarnya sudah
terbukti selamat, dan membuat model hanya belajar dari episode yang kebetulan
dicatat bengkel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

_DAY = np.timedelta64(1, "D")
_TERMINAL_STATUS = set(config.FAILURE_OUTCOME_STATUS) | {config.REPAIR_COMPLETED_STATUS}


def _first_after(times: np.ndarray, journeys: np.ndarray, moment, journey_id: int) -> int:
    """Indeks event pertama yang benar-benar SESUDAH (waktu, journey_id).

    Perbandingan memakai pasangan, bukan waktu saja: beberapa event bisa
    tercatat pada detik yang sama, dan journey_id yang menentukan urutannya.
    """
    low = int(np.searchsorted(times, moment, side="left"))
    high = int(np.searchsorted(times, moment, side="right"))
    within = low + int(np.searchsorted(journeys[low:high], journey_id, side="right"))
    return within if within < high else high


def resolve_outcomes(
    episodes: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, data_end: pd.Timestamp
) -> pd.DataFrame:
    """Tentukan nasib tiap kerusakan, lalu susun bahan mentah fiturnya."""
    episodes = episodes.reset_index(drop=True).copy()
    episodes["failure_onset_on"] = pd.to_datetime(episodes["failure_onset_on"])

    events = events.sort_values(
        ["item_identifier_clean", "created_on", "journey_id"], kind="stable"
    ).reset_index(drop=True)
    event_times = events["created_on"].to_numpy("datetime64[ns]")
    event_journeys = events["journey_id"].to_numpy("int64")
    status = events["status_clean"].fillna("").to_numpy(dtype=object)
    is_failure = events["is_failure_onset"].fillna(False).to_numpy(dtype=bool)
    is_repaired = status == config.REPAIR_COMPLETED_STATUS
    is_installed = status == "INSTALLED"
    is_terminal = np.array([s in _TERMINAL_STATUS for s in status], dtype=bool)
    rows_by_item = events.groupby("item_identifier_clean", sort=False).indices

    total = len(episodes)
    outcome = np.full(total, None, dtype=object)
    reinstalled = np.zeros(total, dtype=bool)
    prior_repaired = np.zeros(total, dtype="int64")
    prior_failures = np.zeros(total, dtype="int64")
    first_seen = np.full(total, np.datetime64("NaT"), dtype="datetime64[ns]")

    onsets = episodes["failure_onset_on"].to_numpy("datetime64[ns]")
    onset_journeys = episodes["onset_journey_id"].to_numpy("int64")

    for position, item in enumerate(episodes["item_identifier_clean"]):
        slot = rows_by_item.get(item)
        if slot is None:
            continue
        times, journeys = event_times[slot], event_journeys[slot]
        moment, journey_id = onsets[position], onset_journeys[position]

        # Riwayat sampai DAN termasuk kerusakan ini - semuanya sudah diketahui
        # pada saat keputusan diambil.
        seen = _first_after(times, journeys, moment, journey_id)
        prior_repaired[position] = int(is_repaired[slot][:seen].sum())
        prior_failures[position] = int(is_failure[slot][:seen].sum())
        first_seen[position] = times[0]

        after_installed = np.flatnonzero(is_installed[slot][seen:])
        after_failure = np.flatnonzero(is_failure[slot][seen:])
        if after_installed.size:
            reinstalled[position] = True

        # Vonis hanya berlaku sampai episode berikutnya dimulai: pemasangan
        # ulang atau kerusakan berikutnya menutup episode ini.
        limits = [times[seen + a[0]] for a in (after_installed, after_failure) if a.size]
        boundary = min(limits) if limits else None

        # Vonis boleh tercatat pada detik yang sama dengan kerusakannya.
        start = int(np.searchsorted(times, moment, side="left"))
        candidates = np.flatnonzero(is_terminal[slot][start:])
        for offset in candidates:
            index = start + offset
            if boundary is not None and times[index] > boundary:
                break
            outcome[position] = status[slot][index]
            break

    episodes["outcome"] = outcome
    episodes["was_reinstalled"] = reinstalled
    episodes["prior_repaired_count"] = prior_repaired
    episodes["prior_failure_count"] = prior_failures
    episodes["age_total_days"] = (onsets - first_seen) / _DAY

    # Umur siklus berjalan: dari pemasangan terakhir sebelum kerusakan ini.
    installs = (
        cycles[["item_identifier_clean", "installed_on"]]
        .assign(installed_on=lambda f: pd.to_datetime(f["installed_on"]))
        .sort_values("installed_on")
    )
    episodes = pd.merge_asof(
        episodes.sort_values("failure_onset_on"),
        installs,
        left_on="failure_onset_on",
        right_on="installed_on",
        by="item_identifier_clean",
        direction="backward",
    ).reset_index(drop=True)
    episodes["cycle_age_days"] = (
        episodes["failure_onset_on"] - episodes["installed_on"]
    ).dt.total_seconds() / 86400.0

    is_scrap = episodes["outcome"].isin(config.SCRAP_STATUS)
    survived = episodes["outcome"].eq(config.REPAIR_COMPLETED_STATUS) | episodes["was_reinstalled"]
    episodes["is_scrap"] = is_scrap.astype(int)
    # Embargo: dekat ujung data, vonis "dibuang" sudah terlihat sementara bukti
    # "diperbaiki" lewat pemasangan ulang belum tentu sempat muncul.
    past_embargo = episodes["failure_onset_on"] <= (
        pd.Timestamp(data_end) - pd.Timedelta(days=config.SCRAP_EMBARGO_DAYS)
    )
    episodes["is_labeled"] = (is_scrap | survived) & past_embargo
    return episodes


def current_state(
    events: pd.DataFrame, cycles: pd.DataFrame, data_end: pd.Timestamp
) -> pd.DataFrame:
    """Kondisi sebuah PART hari ini, dalam bentuk yang sama seperti episode.

    Dipakai prediction: hasilnya dibaca sebagai "seandainya PART ini rusak
    sekarang". Kolom yang dihasilkan sengaja sama persis dengan yang dipakai
    saat training, supaya build_features tidak perlu tahu bedanya.
    """
    if events.empty:
        return pd.DataFrame()

    moment = pd.Timestamp(data_end)
    times = pd.to_datetime(events["created_on"])
    seen = times <= moment
    if not seen.any():
        return pd.DataFrame()

    status = events["status_clean"].fillna("")
    item_type = events.loc[seen, "item_type_clean"].dropna()
    installed = pd.to_datetime(cycles["installed_on"]) if len(cycles) else pd.Series(dtype="datetime64[ns]")
    started = installed[installed <= moment]

    return pd.DataFrame([{
        "item_identifier_clean": events["item_identifier_clean"].iloc[0],
        "failure_onset_on": moment,
        "item_type_clean": item_type.iloc[-1] if len(item_type) else None,
        "age_total_days": (moment - times[seen].min()).total_seconds() / 86400.0,
        "cycle_age_days": (
            (moment - started.max()).total_seconds() / 86400.0 if len(started) else np.nan
        ),
        "prior_repaired_count": int((seen & status.eq(config.REPAIR_COMPLETED_STATUS)).sum()),
        "prior_failure_count": int(
            (seen & events["is_failure_onset"].fillna(False)).sum()
        ),
    }])


def known_item_types(labeled: pd.DataFrame) -> list[str]:
    """Jenis PART yang riwayatnya cukup untuk dipelajari sendiri.

    Dibekukan ke metadata saat training supaya pengelompokan saat prediksi
    persis sama dengan saat model belajar.
    """
    counts = labeled["item_type_clean"].value_counts()
    return sorted(counts[counts >= config.SCRAP_MIN_TYPE_SUPPORT].index)


def build_features(episodes: pd.DataFrame, known_types: list[str]) -> pd.DataFrame:
    """Bangun 7 fitur model scrap dari episode yang sudah dilengkapi riwayat."""
    repaired = pd.to_numeric(episodes["prior_repaired_count"], errors="coerce").fillna(0)
    failures = pd.to_numeric(episodes["prior_failure_count"], errors="coerce").fillna(0)

    features = pd.DataFrame(index=episodes.index)
    features["item_type_category"] = (
        episodes["item_type_clean"]
        .where(episodes["item_type_clean"].isin(known_types), "LOW_SUPPORT")
        .fillna(config.UNKNOWN_LABEL)
        .astype(str)
    )
    # Umur total PART, bukan umur siklus ini saja: PART tua yang baru pertama
    # kali rusak justru yang paling sering langsung dibuang.
    features["log_age_total"] = _log1p(episodes["age_total_days"])
    features["log_cycle_age"] = _log1p(episodes["cycle_age_days"])
    # PART yang pernah berhasil diperbaiki terbukti masih bisa diperbaiki lagi.
    features["log_prior_repaired_count"] = np.log1p(repaired)
    features["has_prior_repair"] = (repaired > 0).astype(float)
    features["log_prior_failure_count"] = np.log1p(failures)
    features["is_first_failure_ever"] = (failures <= 1).astype(float)

    features[config.SCRAP_CATEGORICAL_FEATURES] = features[
        config.SCRAP_CATEGORICAL_FEATURES
    ].astype(str)
    features[config.SCRAP_NUMERIC_FEATURES] = features[
        config.SCRAP_NUMERIC_FEATURES
    ].astype(float)
    return features[config.SCRAP_FEATURE_COLUMNS]


def _log1p(values: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0))
