"""Peluang PART benar-benar MATI (rusak DAN tidak bisa diperbaiki) - gabungan
model failure dan model scrap (Fase B4 restrukturisasi - dedup, sebelumnya
rumus yang sama ditulis terpisah di `predict_scrap.py`, `serving/predictor.py`,
dan `serving/batch_predictor.py`).

Sengaja di `partrisk/` (BUKAN `partrisk/serving/`) walau dipakai serving/ -
`predict_scrap.py` (lapisan ML inti, dipakai berdiri sendiri lewat CLI-nya
sendiri) juga butuh fungsi ini, dan lapisan ML inti tidak boleh bergantung
pada lapisan serving DI ATASnya (lihat docstring api/__init__.py).

    P(mati) = P(rusak dalam horizon) x P(dibuang | rusak)

Sudah dibacktest pada 74.412 observasi: gabungan ini lebih baik daripada
model failure sendirian (PR-AUC naik, 100% dari 500 resampling memihak
gabungan). Tetapi kejadiannya sangat jarang, jadi pakailah sebagai daftar
pantau untuk perencanaan stok - bukan pemicu tindakan per PART.
"""

from __future__ import annotations


def death_probability(failure_probability, scrap_probability):
    """Kedua faktornya sama-sama terkalibrasi, jadi hasil kalinya bisa
    dibaca sebagai perkiraan peluang - dengan catatan: cenderung
    merendahkan (lihat docstring predict_scrap.py soal pergeseran tingkat
    scrap), dan urutannya lebih bisa dipercaya daripada nilainya sendiri.

    Bekerja untuk skalar (satu PART) maupun pandas Series (batch) - `round()`
    Python delegasi ke `Series.__round__` kalau argumennya Series."""
    return round(failure_probability * scrap_probability, 5)
