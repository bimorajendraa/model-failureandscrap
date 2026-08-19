"""Entry point manual: batch prediction untuk seluruh PART aktif.

    python scripts/run_prediction.py
    python scripts/run_prediction.py --output hasil.csv
    python scripts/run_prediction.py --top 20

Memanggil inference.batch_predictor - modul yang SAMA PERSIS dipakai
`GET /api/v1/recommendations` - supaya hasil lewat CLI dan lewat API tidak
mungkin berbeda. Tidak menulis ke database (belum perlu prediction database
untuk tahap ini - lihat README); `--output` opsional hanya menulis CSV lokal
untuk dicek manual.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference import batch_predictor, model_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prediction")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Simpan seluruh hasil ke file CSV (opsional).")
    parser.add_argument("--top", type=int, default=10, help="Berapa baris teratas dicetak.")
    args = parser.parse_args()

    started = time.time()
    logger.info("prediction started")

    try:
        logger.info("model loaded: %s", model_loader.versions())
        scores = batch_predictor.score_active_parts()
    except Exception:
        logger.exception("error saat batch prediction")
        return 1

    frame = scores.frame
    logger.info(
        "prediction completed: %d PART, %d HIGH, %d MEDIUM (%.1f detik)",
        len(frame),
        int(frame["failure_risk_level"].eq("HIGH").sum()),
        int(frame["failure_risk_level"].eq("MEDIUM").sum()),
        time.time() - started,
    )

    columns = [
        "rank", "item_id", "item_type", "failure_risk_level",
        "failure_probability_30d", "scrap_risk_level", "priority", "recommended_action",
    ]
    print(frame[columns].head(args.top).to_string(index=False))

    if args.output:
        frame.to_csv(args.output, index=False)
        logger.info("hasil lengkap disimpan ke %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
