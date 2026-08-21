"""Aplikasi FastAPI untuk predictive maintenance PART.

    uvicorn partrisk.api.main:app --reload

Alur satu request:

    HTTP -> route -> service -> ML core (predict.py / predict_scrap.py)
                                     -> feature_builder -> data_reader -> DB

Client cukup mengirim item_id. Seluruh fitur ML dibangun sendiri oleh ML core
yang sudah ada; tidak ada fitur yang boleh dikirim dari luar.

Aplikasi ini HANYA melayani inference. Training tetap lewat
`python -m partrisk.train` dan `python -m partrisk.train_scrap` - sengaja
tidak ada endpoint untuk melatih model.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from partrisk.api import db_pool, logging_config, settings
from partrisk.api.routes import health, locations, model_info, monitoring, prediction, recommendations
from partrisk.serving import batch_predictor, model_loader
from partrisk.serving.errors import (
    DataSourceUnavailable,
    ModelUnavailable,
    PartNotFound,
    PartNotScorable,
)

logging_config.setup()
logger = logging.getLogger("production_ml.api")

DESCRIPTION = """
API risiko kerusakan dan risiko scrap untuk PART.

**Yang perlu diketahui saat membaca angkanya**

- `failure_probability_Nd` adalah PELUANG PART rusak dalam N hari ke depan.
  Model tidak memperkirakan tanggal kerusakan.
- `scrap_probability` BERSYARAT: peluang PART tidak bisa diperbaiki JIKA
  rusak - bukan peluang PART ini rusak.
- Kelompok risiko (LOW/MEDIUM/HIGH) memakai ambang yang ditetapkan saat
  training dari kapasitas kerja tim, bukan angka bulat yang dikarang.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat model dan siapkan connection pool sekali saat start, bukan
    setiap request."""
    db_pool.install()

    try:
        model_loader.warmup()
        logger.info("Model dimuat: %s", model_loader.versions())
    except ModelUnavailable as error:
        # Aplikasi tetap hidup supaya /health bisa melaporkan keadaannya.
        logger.error("Model production belum tersedia: %s", error)

    if settings.WARMUP_BATCH_ON_STARTUP:
        try:
            scores = batch_predictor.score_active_parts()
            logger.info("Batch scoring awal selesai: %d PART aktif", len(scores.frame))
        except Exception as error:  # noqa: BLE001 - start tidak boleh gagal karenanya
            logger.error("Batch scoring awal gagal: %s", error)

    # Pool database TIDAK ditutup di sini - lihat penjelasan di
    # db_pool.install(). Pada proses production sesungguhnya (satu siklus
    # hidup per proses), OS membereskan soket saat proses keluar.
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description=DESCRIPTION,
    version=health.API_VERSION,
    lifespan=lifespan,
)

# Streamlit tidak memerlukan CORS - panggilannya server-ke-server. Ini untuk
# frontend browser (React/Vue/dll) yang memanggil API langsung; tanpa ini
# browser akan memblokirnya. Default-nya kosong: origin harus disebutkan
# eksplisit lewat CORS_ALLOW_ORIGINS, bukan dibuka untuk semua orang.
if settings.CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        # API ini hanya membaca; tidak ada cookie atau sesi yang perlu dibawa.
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(prediction.router)
app.include_router(recommendations.router)
app.include_router(locations.router)
app.include_router(monitoring.router)


# ---------------------------------------------------------------------------
# Penanganan kesalahan
#
# Client tidak pernah melihat DSN, kredensial, SQL, atau stack trace. Detail
# lengkapnya masuk log server.
# ---------------------------------------------------------------------------


@app.exception_handler(PartNotFound)
async def handle_part_not_found(request: Request, error: PartNotFound) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "status": "NOT_FOUND",
            "item_id": error.item_id,
            "message": error.message,
        },
    )


@app.exception_handler(PartNotScorable)
async def handle_not_scorable(request: Request, error: PartNotScorable) -> JSONResponse:
    """Jaring pengaman.

    Jalur normal sudah menanganinya di route sebagai jawaban 200 berstatus
    NOT_SCORABLE; handler ini hanya untuk jalur yang belum tertangani supaya
    tidak pernah muncul sebagai 500.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "NOT_SCORABLE",
            "item_id": error.item_id,
            "reason": error.reason,
        },
    )


@app.exception_handler(ModelUnavailable)
async def handle_model_unavailable(request: Request, error: ModelUnavailable) -> JSONResponse:
    logger.error("Model tidak tersedia: %s", error)
    return JSONResponse(
        status_code=503,
        content={
            "status": "MODEL_UNAVAILABLE",
            "message": "Model production belum tersedia di server.",
        },
    )


@app.exception_handler(DataSourceUnavailable)
async def handle_data_source(request: Request, error: DataSourceUnavailable) -> JSONResponse:
    logger.exception("Database tidak bisa dibaca")
    return JSONResponse(
        status_code=503,
        content={
            "status": "DATA_SOURCE_UNAVAILABLE",
            "message": "Sumber data sedang tidak bisa dibaca. Coba lagi nanti.",
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
    logger.exception("Kesalahan tak terduga pada %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "status": "INTERNAL_ERROR",
            "message": "Terjadi kesalahan di server.",
        },
    )
