"""Menjaga aplikasi tidak diam-diam memakai angka basi.

`predict.py` menyimpan potret kondisi armada di variabel level-modul dan
mengembalikannya tanpa memeriksa ulang batas waktu data - benar untuk proses
CLI yang hidup sebentar, tapi di server yang hidup berhari-hari begitu
database bertambah, 3 fitur kondisi armada tetap beku di nilai request
pertama sejak start sementara 18 fitur lain sudah segar. Tidak ada error;
prediksi tetap keluar, hanya diam-diam salah.

Modul ini menutup celah itu dari luar tanpa mengubah predict.py: batas waktu
data diperiksa berkala, dan begitu terbukti bergeser, potret armada dibuang
supaya ML core membangunnya ulang dengan pemeriksaannya sendiri. Hasil batch
scoring ikut ditandai basi lewat penanda generation yang sama.
"""

from __future__ import annotations

import logging
import threading
import time

import pandas as pd

import data_reader
import predict as failure_model
from inference import settings

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_data_end: pd.Timestamp | None = None
_checked_at: float = 0.0
# Naik setiap kali data terbukti bertambah. Dipakai batch_predictor untuk tahu
# hasilnya perlu dihitung ulang, tanpa perlu query sendiri.
_generation: int = 0


def current_data_end(force_refresh: bool = False) -> pd.Timestamp:
    """Kejadian terbaru yang tercatat di database.

    Hasilnya ditahan selama DATA_FRESHNESS_TTL_SECONDS: query-nya ringan,
    tetapi dipanggil di setiap request dan nilainya jarang berubah.

    Begitu nilainya terbukti bergeser, potret armada milik ML core dibuang -
    lihat penjelasan di docstring modul.
    """
    global _data_end, _checked_at, _generation

    with _LOCK:
        now = time.time()
        fresh_enough = (
            _data_end is not None
            and now - _checked_at < settings.DATA_FRESHNESS_TTL_SECONDS
        )
        if fresh_enough and not force_refresh:
            return _data_end

        latest = data_reader.get_dataset_max_event_on()
        _checked_at = now

        if _data_end is not None and latest != _data_end:
            logger.info("Data bertambah: %s -> %s. Potret armada dibuang.", _data_end, latest)
            # Membuang potret di ML core, bukan menghitungnya sendiri:
            # pembangunan ulangnya tetap memakai logic dan pemeriksaan milik
            # predict.py, termasuk validasi potret tersimpan terhadap data_end.
            failure_model.clear_fleet_cache()
            _generation += 1

        _data_end = latest
        return _data_end


def generation() -> int:
    """Penanda versi data. Berubah setiap kali database terbukti bertambah."""
    with _LOCK:
        return _generation


def reset() -> None:
    """Lupakan semua yang di-cache. Dipakai test."""
    global _data_end, _checked_at, _generation
    with _LOCK:
        _data_end = None
        _checked_at = 0.0
        _generation = 0
        failure_model.clear_fleet_cache()
