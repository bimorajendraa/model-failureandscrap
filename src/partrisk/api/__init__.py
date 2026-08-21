"""Lapisan serving di atas model yang sudah ada.

Paket ini TIDAK memuat logic machine learning apa pun. Seluruh perhitungan
fitur dan prediksi tetap dikerjakan modul lain di package `partrisk`
(predict.py, predict_scrap.py, feature_builder.py, scrap_features.py,
data_reader.py); di sini hanya dibungkus supaya bisa dipanggil lewat HTTP.
"""

from __future__ import annotations
