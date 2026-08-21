"""Entry point manual: extract -> transform -> feature build, tanpa prediksi.

    python scripts/run_pipeline.py

Membuktikan data pipeline berjalan berdiri sendiri, lepas dari FastAPI/model:

    data_reader      EXTRACT   - SELECT read-only, database -> DataFrame
    feature_builder  TRANSFORM + FEATURE BUILD - DataFrame -> fitur, TANPA
                     menyentuh database sama sekali

Tidak menyimpan apa pun (belum perlu prediction database - lihat README) dan
tidak memuat model. Kalau nanti source database pindah dari local ke server,
hanya `data_reader.connect()`/`config.db_settings()` yang perlu menunjuk ke
tempat baru - urutan di bawah ini tidak berubah.
"""

from __future__ import annotations

import logging
import time

from partrisk import data_reader
from partrisk.features import failure as feature_builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def main() -> int:
    started = time.time()
    logger.info("pipeline started")

    try:
        cycles = data_reader.get_cycles()
        events = data_reader.get_events()
        episodes = data_reader.get_failure_episodes()
        logger.info("database connected")
        logger.info(
            "rows extracted: %d siklus, %d event, %d kerusakan",
            len(cycles), len(events), len(episodes),
        )

        # TRANSFORM: siklus mentah -> satu snapshot per PART yang sedang aktif.
        observations = feature_builder.current_observations(cycles)
        logger.info("rows transformed: %d PART aktif", len(observations))

        # FEATURE BUILD: tempelkan riwayat + kondisi armada - kolom yang sama
        # persis dipakai predict.py dan batch scoring, feature_builder tidak
        # pernah menyentuh database sendiri.
        observations = feature_builder.attach_history(observations, events)
        observations = feature_builder.attach_fleet(observations, cycles, episodes)
        logger.info("features generated: %d kolom", len(observations.columns))
    except Exception:
        logger.exception("error saat menjalankan pipeline")
        return 1

    logger.info("pipeline selesai dalam %.1f detik", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
