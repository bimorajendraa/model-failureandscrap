"""Rincian event mentah di balik faktor risiko yang berupa hitungan.

`explanation.py` meringkas riwayat PART menjadi kalimat seperti "2 kerusakan
tercatat dalam 365 hari terakhir" - benar, tapi tidak bisa dijawab "kapan
saja itu?" tanpa membuka datanya. Modul ini menjawab itu: mengambil baris
event asli (dari `data_reader.get_events`, ML core yang sama, tanpa hitungan
ulang) dan menyusunnya jadi tabel tanggal.

Ini BUKAN fitur model - tidak dipakai `predict()` atau `predict_scrap()`
sama sekali. Ini murni untuk memberi konteks bagi manusia yang membaca faktor
risiko di halaman detail.
"""

from __future__ import annotations

import pandas as pd


def failure_history(events: pd.DataFrame) -> list[dict]:
    """Tanggal setiap kerusakan yang tercatat, terbaru dulu.

    `is_failure_onset` adalah definisi kerusakan yang sama persis dipakai
    feature_builder.py untuk menghitung `prior_failure_count` dkk - jadi
    jumlah baris di sini selalu cocok dengan angka yang ditampilkan sebagai
    faktor risiko.
    """
    failures = events.loc[events["is_failure_onset"].fillna(False).astype(bool)].copy()
    if failures.empty:
        return []
    failures = failures.sort_values("created_on", ascending=False)
    return [
        {
            "date": str(pd.Timestamp(row["created_on"])),
            "location": (
                row["place_canonical_clean"]
                if pd.notna(row["place_canonical_clean"])
                else None
            ),
            "status": row["status_clean"],
        }
        for _, row in failures.iterrows()
    ]


def location_history(events: pd.DataFrame) -> list[dict]:
    """Lokasi yang pernah tercatat, dengan rentang tanggal terlihat di sana.

    Diurutkan dari yang paling belakangan aktif - itu yang paling relevan
    untuk pertanyaan "sekarang ada di mana / terakhir di mana".
    """
    known = events.loc[events["place_canonical_clean"].notna()].copy()
    if known.empty:
        return []
    known["created_on"] = pd.to_datetime(known["created_on"])
    grouped = known.groupby("place_canonical_clean")["created_on"].agg(
        first_seen="min", last_seen="max", events="count"
    )
    grouped = grouped.sort_values("last_seen", ascending=False)
    return [
        {
            "location": location,
            "first_seen": str(row["first_seen"]),
            "last_seen": str(row["last_seen"]),
            "events": int(row["events"]),
        }
        for location, row in grouped.iterrows()
    ]
