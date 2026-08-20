"""Konteks instalasi (item type & lokasi) PERSIS pada installed_on.

`data_reader.get_events()` (yang SUDAH dibaca `build_dataset.build()`)
mengembalikan `item_type_clean` dan `place_canonical_clean` yang SUDAH
dikanonikalisasi `data_reader.py` untuk SEMUA event, termasuk baris
`status_clean=='INSTALLED'` yang membuka tiap lifecycle. Modul ini hanya
MENJOIN baris itu ke lifecycle - TIDAK ada query baru, TIDAK ada mapping/
kanonikalisasi baru.

`device_type`/`device_model` SENGAJA tidak dibuat: `item_category` pada
cohort survival selalu 'PART' (satu-satunya kategori lain adalah 'TERMINAL'
tanpa relasi ybs yang sudah dikanonikalisasi data_reader.py) - mengekstraknya
butuh JOIN PART->TERMINAL baru yang berarti mapping baru, di luar cakupan
"reuse kolom canonical yang sudah ada". Didokumentasikan sebagai keterbatasan
data di README, bukan dipaksakan.
"""

from __future__ import annotations

import pandas as pd

UNKNOWN_LABEL = "UNKNOWN"


def attach_install_context(observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan item_type_at_install & place_at_install ke tiap lifecycle.

    Join by (item_identifier_clean, installed_on) ke event INSTALLED yang
    membuka siklus itu. Diverifikasi bersih pada data produksi: 24.045 cycles
    -> 24.045 baris hasil join (tidak ada penggandaan baris), sesudah
    drop_duplicates menangani 8 baris event ber-timestamp kembar (4 pasang,
    diselesaikan ambil yang pertama - dilaporkan di data_validation, bukan
    didiamkan).
    """
    installed_events = (
        events.loc[
            events["status_clean"].eq("INSTALLED"),
            ["item_identifier_clean", "created_on", "item_type_clean", "place_canonical_clean"],
        ]
        .drop_duplicates(subset=["item_identifier_clean", "created_on"], keep="first")
    )
    merged = observations.merge(
        installed_events,
        left_on=["item_identifier_clean", "installed_on"],
        right_on=["item_identifier_clean", "created_on"],
        how="left",
    )
    merged["item_type_at_install"] = merged["item_type_clean"].fillna(UNKNOWN_LABEL).astype(str)
    merged["place_at_install"] = merged["place_canonical_clean"].fillna(UNKNOWN_LABEL).astype(str)
    return merged.drop(columns=["created_on", "item_type_clean", "place_canonical_clean"])
