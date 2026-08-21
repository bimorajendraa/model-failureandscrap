"""Lokasi file di disk - model, .env, dan root package itu sendiri."""

from __future__ import annotations

import os
from pathlib import Path

# config/paths.py ada di src/partrisk/config/paths.py - repo root (tempat
# models/ dan .env sungguhan tinggal, models/ ARTIFACT bukan bagian package)
# ada EMPAT tingkat di atasnya (paths.py -> config/ -> partrisk/ -> src/ ->
# root). Default ini menghitung itu secara struktural, BUKAN menebak - jadi
# benar otomatis selama layout src/partrisk/config/ ini dipertahankan, tanpa
# perlu env var apa pun di dev biasa. PARTRISK_HOME override tetap ada untuk
# kasus di mana struktur relatif itu TIDAK berlaku (mis. Docker image yang
# tidak menyalin src/ apa adanya - lihat Dockerfile ENV PARTRISK_HOME).
PACKAGE_DIR = Path(os.environ.get("PARTRISK_HOME", str(Path(__file__).resolve().parent.parent.parent.parent)))
MODEL_DIR = Path(os.environ.get("PARTRISK_MODEL_DIR", str(PACKAGE_DIR / "models")))
ENV_FILE = Path(os.environ.get("PARTRISK_ENV_FILE", str(PACKAGE_DIR / ".env")))
# Satu folder per model, masing-masing berisi CURRENT + v1, v2, ... supaya
# tidak ada dua "v1" yang artinya berbeda.
FAILURE_MODEL_DIR = MODEL_DIR / "failure"
SCRAP_MODEL_DIR = MODEL_DIR / "scrap"
