"""Kredensial database."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from partrisk.config.paths import ENV_FILE


def db_settings() -> dict[str, str]:
    """Kredensial database dari .env / environment. Production hanya membaca."""
    load_dotenv(ENV_FILE)
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Konfigurasi database belum lengkap: "
            + ", ".join(missing)
            + ". Salin .env.example menjadi .env lalu isi nilainya."
        )
    return {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "sslmode": os.getenv("DB_SSLMODE", "prefer"),
    }
