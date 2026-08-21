"""Bangun observasi landmark (Tahap 6-9, "event-based survival"): BANYAK
titik observasi per lifecycle, bukan satu (installed_on) seperti model
statis di `survival_model/` induk.

Objective beda dari model statis: "dengan kondisi PART SAAT INI (umur A,
riwayat sampai titik ini), berapa lama lagi sampai failure?" - bukan "saat
pertama dipasang, berapa lama PART akan bertahan?".

## Desain landmark: titik mana yang jadi observasi

Audit data (lihat reports/intra_cycle_event_audit.md) menemukan 80,3% dari
23.927 lifecycle TIDAK punya event operasional sama sekali di antara
INSTALLED dan akhir siklus - kebalikan dari asumsi "banyak event menandai
perubahan kondisi". Jadi landmark di sini adalah GABUNGAN tiga sumber, bukan
murni event organik:

1. **L=installed_on** (age=0) - SELALU ada, ekuivalen dengan satu-satunya
   observasi model statis, supaya event-based tetap punya baseline yang
   bisa dibandingkan head-to-head (lihat evaluate.py t0_only_c_index).
2. **Event organik** - operational event APA PUN pada item yang sama,
   STRICTLY di antara installed_on dan endpoint lifecycle (failure_onset_on
   kalau event, cutoff_on kalau censored) - kalau ada (~20% lifecycle).
3. **Anchor jarang** (90, 180, 365 hari, lalu +365 hari, dibatasi
   `MAX_ANCHORS_PER_LIFECYCLE`) - SENGAJA bukan grid 30-harian tetap: cycle
   yang bertahan bertahun-tahun (umum di TRAIN, installed_on bisa 2014)
   akan menghasilkan puluhan snapshot redundant kalau anchornya rapat -
   persis pola classification grid yang ingin dihindari sejak awal
   eksperimen survival ini. Interval MELEBAR (bukan tetap) supaya densitas
   anchor tinggi di awal umur (informasi paling berharga) dan menurun untuk
   ekor lifecycle yang sangat panjang.

## Split & cutoff: mengikuti LIFECYCLE, bukan L masing-masing

Keputusan desain PALING PENTING di modul ini: split (TRAIN/VALIDATION/TEST)
dan cutoff administrative censoring sebuah landmark row ditentukan oleh
lifecycle induknya (installed_on, SAMA PERSIS dengan model statis lewat
`lifecycle.assign_lifecycle_outcome()`), BUKAN dihitung ulang dari L
masing-masing.

Alternatif yang DITOLAK: assign split per-L (misal cycle yang installed_on
2020 dan masih aktif bisa punya landmark early di TRAIN, landmark 2025 di
VALIDATION, landmark 2026 di TEST). Itu tidak menghasilkan leakage temporal
(tiap L tetap hanya memakai fitur/label sampai L), TAPI membuat SATU
lifecycle fisik (kadang SATU item_identifier_clean) muncul di TRAIN *dan*
VALIDATION/TEST via landmark yang berbeda - model bisa "mengenali" identitas
sebuah item lewat kombinasi fitur unik lintas landmark, bukan cuma belajar
generalisasi. README model statis SUDAH mendokumentasikan risiko serupa
(7,5% item beririsan split lewat previous-cycle feature) sebagai leakage
non-temporal yang diterima - landmark per-L akan MEMPERBESAR risiko itu
drastis (lebih banyak lifecycle yang landmark-nya menyeberang split, by
construction). Desain di sini (split ikut lifecycle) MENGHILANGKAN risiko
itu sepenuhnya, dengan konsekuensi: lifecycle TRAIN yang installed_on-nya
lama (misal 2014) tetap boleh menghasilkan landmark sampai validation_start
(2025) - umurnya panjang tapi TIDAK menyeberang ke VALIDATION/TEST.

## Reuse total logika censoring - TIDAK ada aturan baru

`event_observed`/`duration_days` (dari install) untuk sebuah lifecycle
SUDAH final dari `lifecycle.assign_lifecycle_outcome()` (diimpor
APA ADANYA, tidak diubah). Landmark HANYA menggeser titik ASAL pengukuran
durasi dari installed_on ke L:

    age_at_landmark    = L - installed_on
    duration_landmark  = duration_days - age_at_landmark   (residual)
    event_landmark     = event_observed                    (tidak berubah)

berlaku selama `age_at_landmark < duration_days` (L terjadi sebelum endpoint
lifecycle - kalau tidak, L bukan landmark yang valid, dibuang). Tidak ada
percabangan FAILURE/CENSORED/EXCLUDE baru yang ditulis di sini - itu
sepenuhnya keputusan `lifecycle.py` yang sudah diaudit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAY = np.timedelta64(1, "D")
ANCHOR_BASE_AGES_DAYS = (90.0, 180.0, 365.0)
ANCHOR_STEP_DAYS = 365.0
MAX_ANCHORS_PER_LIFECYCLE = 8
MAX_ORGANIC_PER_LIFECYCLE = 8  # jaring pengaman - lihat catatan di build_landmarks()


def _anchor_ages(max_age_days: float, max_anchors: int = MAX_ANCHORS_PER_LIFECYCLE) -> list[float]:
    """Umur anchor (hari sejak installed_on) untuk satu lifecycle: 90/180/365
    lalu +365 tiap langkah, dibatasi `max_anchors` DAN harus < max_age_days
    (lifecycle harus masih berjalan pada umur itu)."""
    ages = list(ANCHOR_BASE_AGES_DAYS)
    while len(ages) < max_anchors:
        ages.append(ages[-1] + ANCHOR_STEP_DAYS)
    return [a for a in ages if a < max_age_days]


def build_landmarks(outcome: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """`outcome` = HASIL `lifecycle.assign_lifecycle_outcome()`
    (SUDAH punya split/cutoff_on/duration_days/event_observed/eligible per
    lifecycle, tidak diubah di sini). `events` = `data_reader.get_events()`.

    Kembalikan satu baris per (lifecycle, landmark) - kolom identitas
    lifecycle asli (installation_cycle_id, item_identifier_clean,
    item_model_code_clean, installed_client_clean, split, cutoff_on)
    ditambah `observation_on` (=L), `landmark_age_days` (=L-installed_on,
    SAMA dengan days_since_installation setelah _finalize di
    feature_builder), `duration_days`/`event_observed` residual dari L, dan
    `landmark_source` (INSTALL/ORGANIC_EVENT/ANCHOR - diagnostik, TIDAK
    dipakai sebagai fitur model)."""
    eligible = outcome.loc[outcome["eligible"]].reset_index(drop=True)

    events_sorted = events.sort_values(["item_identifier_clean", "created_on"], kind="stable")
    event_times_by_item = {
        item: sub["created_on"].to_numpy("datetime64[ns]")
        for item, sub in events_sorted.groupby("item_identifier_clean", sort=False)
    }

    installed = eligible["installed_on"].to_numpy("datetime64[ns]")
    duration = eligible["duration_days"].to_numpy(dtype=float)
    items = eligible["item_identifier_clean"].to_numpy()

    rows_ages: list[np.ndarray] = []
    rows_source: list[np.ndarray] = []
    rows_index: list[np.ndarray] = []

    for i in range(len(eligible)):
        max_age = duration[i]
        ages = [0.0]
        sources = ["INSTALL"]

        item_times = event_times_by_item.get(items[i])
        if item_times is not None and len(item_times):
            install_i = installed[i]
            end_i = install_i + np.timedelta64(int(round(max_age)), "D")
            # STRICT: created_on > installed_on dan < endpoint lifecycle -
            # event tepat DI installed_on/endpoint bukan landmark baru
            # (sudah tercakup L=0 / adalah endpoint itu sendiri).
            in_window = item_times[(item_times > install_i) & (item_times < end_i)]
            if len(in_window):
                organic_ages = (in_window - install_i) / _DAY
                # Bulatkan ke hari bulat (konsisten dengan duration_days) +
                # dedup - beberapa event operasional (REQUESTED/ISSUED/
                # DELIVERY) sering terjadi di hari yang sama.
                organic_ages = np.unique(np.round(organic_ages))
                organic_ages = organic_ages[(organic_ages > 0) & (organic_ages < max_age)]
                if len(organic_ages) > MAX_ORGANIC_PER_LIFECYCLE:
                    # Jaring pengaman untuk kasus langka (maks teramati 16,
                    # lihat reports/intra_cycle_event_audit.md) - ambil yang
                    # PALING BARU (paling relevan dengan kondisi mendekati
                    # endpoint), bukan yang paling awal.
                    organic_ages = organic_ages[-MAX_ORGANIC_PER_LIFECYCLE:]
                ages.extend(organic_ages.tolist())
                sources.extend(["ORGANIC_EVENT"] * len(organic_ages))

        for age in _anchor_ages(max_age):
            ages.append(age)
            sources.append("ANCHOR")

        ages_arr = np.round(np.asarray(ages, dtype=float))
        # Dedup FINAL lintas ketiga sumber (anchor bisa kebetulan sama
        # dengan hari event organik) - urutan prioritas INSTALL >
        # ORGANIC_EVENT > ANCHOR dipertahankan lewat pd.Series.duplicated
        # setelah sort stabil by ages lalu prioritas.
        priority = {"INSTALL": 0, "ORGANIC_EVENT": 1, "ANCHOR": 2}
        order = sorted(range(len(ages_arr)), key=lambda k: (ages_arr[k], priority[sources[k]]))
        seen: set[float] = set()
        keep_idx: list[int] = []
        for k in order:
            if ages_arr[k] not in seen:
                seen.add(ages_arr[k])
                keep_idx.append(k)

        final_ages = ages_arr[keep_idx]
        final_sources = np.asarray(sources)[keep_idx]
        # age harus STRICT < duration (residual harus positif, >=1 hari -
        # konsisten dengan invarian duration_days >= 1.0 di lifecycle.py).
        valid = final_ages < max_age
        rows_ages.append(final_ages[valid])
        rows_source.append(final_sources[valid])
        rows_index.append(np.full(valid.sum(), i))

    landmark_age = np.concatenate(rows_ages)
    landmark_source = np.concatenate(rows_source)
    source_row = np.concatenate(rows_index)

    landmarks = eligible.iloc[source_row].reset_index(drop=True)
    landmarks["landmark_age_days"] = landmark_age
    landmarks["landmark_source"] = landmark_source
    landmarks["observation_on"] = (
        landmarks["installed_on"].to_numpy("datetime64[ns]")
        + landmark_age.astype("timedelta64[D]")
    )
    landmarks["duration_days"] = landmarks["duration_days"].to_numpy(dtype=float) - landmark_age
    # event_observed TIDAK berubah (lihat docstring modul) - disalin apa
    # adanya dari lifecycle induk untuk kejelasan (bukan dihitung ulang).

    return landmarks
