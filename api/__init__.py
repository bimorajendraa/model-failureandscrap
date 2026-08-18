"""Lapisan serving di atas model yang sudah ada.

Paket ini TIDAK memuat logic machine learning apa pun. Seluruh perhitungan
fitur dan prediksi tetap dikerjakan modul di root repository (predict.py,
predict_scrap.py, feature_builder.py, scrap_features.py, data_reader.py);
di sini hanya dibungkus supaya bisa dipanggil lewat HTTP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Modul ML ada di root repository, satu tingkat di atas paket ini. Ditambahkan
# eksplisit supaya `uvicorn api.main:app` tetap jalan dari direktori mana pun,
# bukan hanya dari root.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
