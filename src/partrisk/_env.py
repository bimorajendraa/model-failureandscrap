"""Helper baca environment variable kecil, dipakai `api/settings.py` dan
`serving/settings.py` (Fase B4 restrukturisasi - dedup, sebelumnya masing-
masing punya salinan sendiri yang identik)."""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default
