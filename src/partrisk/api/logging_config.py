"""Konfigurasi logging - dipanggil sekali saat aplikasi start.

Modul-modul di api/ sudah memanggil logging.getLogger(__name__) dan mencatat
kejadian penting (model dimuat, batch scoring selesai, potret armada dibuang),
tapi tanpa logging dikonfigurasi, pesan level INFO hilang diam-diam - Python
hanya punya handler darurat untuk WARNING ke atas. Tanpa modul ini, startup
production terlihat sukses tanpa jejak apakah model benar-benar dimuat.
"""

from __future__ import annotations

import logging
import os
import sys

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

    _configured = True
