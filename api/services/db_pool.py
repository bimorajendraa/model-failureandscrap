"""Connection pooling untuk data_reader.connect(), TANPA mengubah data_reader.py.

data_reader.connect() membuka satu koneksi baru per panggilan - benar untuk
predict.py/train.py yang berjalan sebentar, tapi boros di API yang melayani
banyak request bersamaan. Modul ini menambal data_reader.connect SEKALI saat
API start (pola yang sama dengan query_cache.py) supaya seluruh pemanggilan
`with data_reader.connect() as conn:` yang sudah ada transparan memakai
koneksi dari pool, tanpa mengubah satu baris pun kode ML core.

Hanya API yang memasangnya, lewat install() di lifespan main.py. Dipanggil
dari terminal (predict.py/train.py), data_reader.connect tetap
psycopg.connect() langsung apa adanya.
"""

from __future__ import annotations

import atexit
import logging
import threading

import psycopg_pool

import config
import data_reader

logger = logging.getLogger(__name__)

_pool: psycopg_pool.ConnectionPool | None = None
_lock = threading.Lock()

# Kecil dengan sengaja: aplikasi ini melayani dashboard internal, bukan lalu
# lintas publik. Batch scoring dan satu assessment masing-masing memakai
# paling banyak 1 koneksi pada satu waktu (lihat query_cache.py), jadi pool
# sekecil ini sudah menutupi beberapa request bersamaan.
MIN_SIZE = 1
MAX_SIZE = 8


def install() -> None:
    """Buat pool sekali per proses dan tambal data_reader.connect.

    Idempoten seperti query_cache.install() - bukan "tutup lalu buat ulang":
    test suite membuat beberapa TestClient terpisah dalam satu proses, dan
    "tutup lalu buat ulang" pada tiap start aplikasi berarti TestClient kedua
    menutup pool yang masih dipakai yang pertama. Production sesungguhnya
    hanya punya satu siklus hidup aplikasi per proses, jadi idempoten ini
    juga yang benar di sana.
    """
    global _pool
    with _lock:
        if _pool is not None and not _pool.closed:
            return

        _pool = psycopg_pool.ConnectionPool(
            conninfo="",
            kwargs={
                **config.db_settings(),
                "application_name": "production_ml_api",
                "options": "-c default_transaction_read_only=on",
            },
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            open=True,
        )
        data_reader.connect = _pool.connection
        logger.info("Connection pool database siap (min=%d, max=%d)", MIN_SIZE, MAX_SIZE)

        # Sekali per proses (bukan per siklus hidup aplikasi), lewat atexit
        # daripada __del__ garbage collector - __del__ bisa terpanggil setelah
        # stdout sudah ditutup dan gagal saat mencoba logging.
        atexit.register(teardown)


def teardown() -> None:
    """Tutup pool secara eksplisit.

    Tidak dipanggil otomatis saat aplikasi berhenti (lihat install()) -
    dipanggil lewat atexit sekali saat interpreter keluar, dan tersedia untuk
    pembersihan manual di luar itu.
    """
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
            # Tidak logging di sini: pada saat atexit berjalan, stream
            # logging bisa saja sudah ditutup duluan oleh mekanisme lain.


def stats() -> dict:
    """Statistik pool untuk /health - kosong kalau belum dipasang."""
    if _pool is None:
        return {"installed": False}
    info = _pool.get_stats()
    return {
        "installed": True,
        "pool_min": MIN_SIZE,
        "pool_max": MAX_SIZE,
        "connections_in_use": info.get("pool_size", 0) - info.get("pool_available", 0),
        "connections_idle": info.get("pool_available", 0),
    }
