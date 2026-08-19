"""Connection pooling untuk data_reader.connect(), TANPA mengubah data_reader.py.

MASALAH YANG DIPECAHKAN DI SINI

`data_reader.connect()` membuka SATU koneksi psycopg baru per panggilan -
benar untuk predict.py/train.py yang berjalan sebagai proses CLI sekali
pakai, tetapi di API yang melayani banyak request bersamaan, membuka koneksi
TCP+autentikasi baru setiap kali `api/services/query_cache.py` tidak bisa
menyatukan pembacaan (mis. dua request dari PART berbeda datang bersamaan)
itu boros dan bisa membebani database saat trafik naik.

Modul ini menambal `data_reader.connect` SEKALI saat API start - persis pola
yang sudah dipakai `query_cache.py` untuk menyatukan `get_events`/`get_cycles`
- supaya seluruh pemanggilan `with data_reader.connect() as conn:` yang sudah
ada (termasuk di dalam data_reader.py sendiri) transparan memakai koneksi
dari pool, tanpa mengubah satu baris pun kode ML core.

BATASNYA JELAS

- Hanya API yang memasangnya (lewat install() di lifespan main.py).
  `python predict.py` atau `python train.py` dari terminal tetap memakai
  psycopg.connect() langsung apa adanya - tidak pernah menyentuh pool ini.
- Semantik read-only dan timeout yang sama persis dengan connect() asli
  dipertahankan (options=default_transaction_read_only=on, application_name).
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
    """Buat pool sekali per proses dan tambal data_reader.connect. Aman
    dipanggil berulang - panggilan kedua dan seterusnya tidak melakukan
    apa-apa selama pool yang ada masih terbuka.

    SENGAJA idempoten seperti query_cache.install(), bukan "tutup lalu buat
    ulang" seperti versi sebelumnya: test suite membuat banyak TestClient
    terpisah dalam satu proses yang sama (lihat tests/test_dashboard.py,
    tests/test_api.py) - "tutup lalu buat ulang" pada setiap start aplikasi
    berarti TestClient KEDUA menutup pool yang masih dipakai TestClient
    PERTAMA (dan pemanggilan data_reader langsung di luar TestClient mana
    pun, seperti di conftest.py), meledakkan PoolClosed di tengah test lain
    yang sedang berjalan. Dalam production sesungguhnya hanya ada SATU
    siklus hidup aplikasi per proses, jadi idempoten ini juga yang benar
    di sana.
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

        # Sekali per PROSES (bukan per siklus hidup aplikasi - lihat
        # docstring install()), supaya pool ditutup rapi lewat close() saat
        # interpreter benar-benar keluar, alih-alih dibiarkan diserahkan ke
        # __del__ garbage collector yang bisa mencoba logging setelah stdout
        # sudah ditutup duluan.
        atexit.register(teardown)


def teardown() -> None:
    """Tutup pool secara eksplisit.

    TIDAK dipanggil otomatis saat aplikasi berhenti (lihat penjelasan di
    install()) - dipanggil lewat atexit SEKALI saat interpreter benar-benar
    keluar, dan tersedia untuk pembersihan manual di luar itu.
    """
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
            # TIDAK logging di sini: dipanggil lewat atexit, dan pada tahap
            # itu stream logging bisa saja sudah ditutup duluan oleh
            # mekanisme lain (mis. capture pytest saat test selesai) - logging
            # module sendiri yang mencetak diagnostik kegagalannya langsung
            # ke stderr saat itu terjadi, di luar jangkauan try/except di
            # sini. Bukan kegagalan sungguhan, hanya urutan shutdown yang di
            # luar kendali - lebih baik tidak mencoba logging sama sekali.


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
