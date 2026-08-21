"""Konteks device/terminal (Tahap 2/12) - PART -> TERMINAL parent link,
KONSTAN per lifecycle (bicara tentang device tempat PART dipasang, bukan
kondisi berjalan siklus).

Data dari `data_reader.get_terminal_context()` - query kanonikal BARU yang
dibangun ulang dari tabel mentah (`journal.t_item_request_out`,
`master.t_mtr_item`, `inventory.t_item`), TIDAK bergantung pada schema
`analytics` (lihat docstring lengkap di data_reader.py - direproduksi dari
definisi VIEW riset `analytics.eda_part_terminal_cycle_link` lewat
`pg_get_viewdef`, diverifikasi angkanya PERSIS sama: 24.008/24.045 valid
link, 10.313 baris "recorded after installation").

Point-in-time safety: HANYA baris `parent_link_quality_status ==
'VALID_POINT_IN_TIME_RELATION'` yang dipakai sebagai terminal_type/model -
selain itu (relasi baru diketahui SETELAH instalasi, atau tidak ketemu sama
sekali) diberi UNKNOWN_LABEL, BUKAN diam-diam dipakai seolah sudah diketahui
sejak awal.
"""

from __future__ import annotations

import pandas as pd

UNKNOWN_LABEL = "UNKNOWN"


def attach_terminal_context(observations: pd.DataFrame, terminal_raw: pd.DataFrame) -> pd.DataFrame:
    """Tempelkan `terminal_type_context`/`terminal_model_context` ke tiap
    lifecycle. `terminal_raw` = hasil `data_reader.get_terminal_context()`
    APA ADANYA (satu baris per event INSTALLED - dedup di sini kalau ada
    timestamp kembar, pola sama dengan `install_context.py`).

    Join by (item_identifier_clean, installed_on) - SAMA PERSIS dengan
    `install_context.attach_install_context()`.
    """
    safe = terminal_raw.loc[terminal_raw["parent_link_quality_status"].eq("VALID_POINT_IN_TIME_RELATION")].copy()
    safe = safe.drop_duplicates(subset=["item_identifier_clean", "installed_on"], keep="first")

    merged = observations.merge(
        safe[["item_identifier_clean", "installed_on", "terminal_type_clean", "terminal_model_code_clean"]],
        on=["item_identifier_clean", "installed_on"], how="left",
    )
    merged["terminal_type_context"] = merged["terminal_type_clean"].fillna(UNKNOWN_LABEL).astype(str)
    merged["terminal_model_context"] = merged["terminal_model_code_clean"].fillna(UNKNOWN_LABEL).astype(str)
    return merged.drop(columns=["terminal_type_clean", "terminal_model_code_clean"])
