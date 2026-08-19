"""Inferensi: model + fitur -> hasil prediksi. Independen dari FastAPI.

Paket ini bisa dipakai dari mana pun (script CLI, test, API) tanpa perlu
server HTTP hidup - hanya membungkus predict.py/predict_scrap.py/
feature_builder.py yang sudah ada di root repository, tidak menghitung ulang
logic apa pun.

    model_loader     muat model + metadata sekali per proses
    predictor        prediksi satu PART
    batch_predictor  prediksi seluruh PART aktif sekaligus (vectorized)
    recommendation   terjemahkan risiko jadi tindakan operasional
    explanation      faktor risiko dalam bahasa manusia
    history          riwayat kerusakan/lokasi mentah untuk halaman detail
"""

from __future__ import annotations

import sys
from pathlib import Path

# Modul ML ada di root repository, satu tingkat di atas paket ini - sama
# seperti api/__init__.py.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
