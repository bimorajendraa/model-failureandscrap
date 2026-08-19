"""Konfigurasi logging - dipanggil sekali saat aplikasi start.

MASALAH YANG DIPECAHKAN DI SINI

Modul-modul di api/ sudah memanggil `logging.getLogger(__name__)` dan mencatat
kejadian penting (model dimuat, batch scoring selesai, potret armada dibuang
karena data bertambah, kegagalan geocoding, database tidak bisa dibaca) -
tetapi tanpa logging DIKONFIGURASI, pesan level INFO itu hilang diam-diam.
Python hanya punya "handler darurat" untuk WARNING ke atas; di bawah itu,
tidak ada yang tercetak sama sekali kecuali root logger diberi handler.

Akibatnya di production: startup terlihat sukses, tapi tidak ada jejak APAKAH
model benar-benar dimuat, batch scoring awal berhasil, atau data ternyata
sudah bertambah dan potret armada dibuang. Semuanya sudah dicatat lewat
logger.info() - hanya tidak pernah keluar ke mana pun.
"""

from __future__ import annotations

import logging
import os
import sys

# Format terstruktur satu baris: waktu, level, nama logger, pesan. Cukup
# untuk dibaca manusia langsung DAN cukup teratur untuk di-parse alat log
# (mis. dipecah per kolom oleh spasi/kurung siku) tanpa perlu dependensi JSON
# logging tambahan - sesuai skala aplikasi ini sekarang.
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup() -> None:
    """Pasang logging root. Aman dipanggil berulang (idempotent)."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # uvicorn punya logger sendiri (uvicorn.error, uvicorn.access) yang sudah
    # dikonfigurasi uvicorn sendiri saat dijalankan lewat CLI `uvicorn ...` -
    # tidak disentuh di sini supaya tidak mendobel baris log saat itu terjadi.
    # Saat dijalankan cara lain (mis. lewat TestClient di test), logger uvicorn
    # ikut memakai root handler di atas - tetap konsisten.

    _configured = True
