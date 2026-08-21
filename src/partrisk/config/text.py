"""Kanonikalisasi teks (client/lokasi).

Mapping yang sudah disetujui reviewer pada fase research. Disimpan sebagai
konstanta supaya production tidak bergantung pada tabel di schema analytics.
"""

from __future__ import annotations

APPROVED_LOCATION_ALIAS = {"GUDANG NUTECH": "GUDANG NI"}
APPROVED_CLIENT_ALIAS: dict[str, str] = {}
TEXT_ABBREVIATION_MAPPING = {"JKT": "JAKARTA"}

# Kandidat fuzzy diterima otomatis hanya kalau sangat mirip DAN jauh lebih
# mirip dibanding kandidat kedua.
FUZZY_MIN_SCORE = 0.90
FUZZY_MIN_MARGIN = 0.08
