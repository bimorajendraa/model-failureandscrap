"""Koordinat lokasi lewat OpenStreetMap Nominatim, dengan penyaringan ketat.

Nama lokasi di database ini bukan alamat lengkap - hanya nama singkat
("STASIUN JUANDA", "GUDANG NI") - dan geocoding otomatis polos untuk nama
sesingkat itu TERBUKTI berbahaya: dicoba langsung, "SERVICE CENTER" memang
ketemu, tapi nyangkut ke bangunan retail di Semarang, bukan gudang servis di
Jakarta. Pin yang salah tempat lebih menyesatkan daripada tidak ada pin sama
sekali untuk keputusan operasional - jadi hasil disaring ketat:

- HANYA diterima kalau koordinatnya jatuh di dalam kotak Jabodetabek. Ini
  bukan tebakan sembarangan: seluruh client yang tercatat di data (KCI, LRT
  Jabodebek, Railink bandara) beroperasi di situ, jadi kotak ini adalah batas
  geografis yang didukung data itu sendiri, bukan angka yang dikarang.
- Lokasi yang gagal lolos saringan TIDAK ditampilkan sebagai pin - dilaporkan
  terpisah sebagai "belum punya koordinat", supaya petanya jujur tentang apa
  yang tidak diketahuinya.

Hasilnya di-cache di disk (`.cache/geocode.json`): nama lokasi tidak berubah
dari hari ke hari, jadi tidak perlu digeocode ulang setiap kali peta dibuka.
Mematuhi kebijakan pemakaian Nominatim: User-Agent deskriptif, maksimum satu
permintaan per detik, dan hanya untuk lokasi yang belum ada di cache.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import requests

from api import ROOT_DIR

logger = logging.getLogger(__name__)

CACHE_PATH = ROOT_DIR / ".cache" / "geocode.json"

# Kotak pembatas Jabodetabek (Jakarta, Bogor, Depok, Tangerang, Bekasi) plus
# sedikit ruang - mencakup seluruh area operasi client yang tercatat di data,
# termasuk bandara Soekarno-Hatta. Diturunkan dari cakupan operasi client yang
# ada di data, bukan angka bulat yang dikarang.
#
# Batas barat SENGAJA dipepetkan ke 106.23, bukan dibulatkan lebar: cukup
# untuk mencakup Rangkasbitung (koordinat asli 106.2516, ujung jalur KRL
# Commuter Line yang batas lama 106.30 salah membuangnya), tetapi tetap
# membuang stasiun jalur Merak yang bukan Commuter Line walau nama tempatnya
# juga muncul di data dan geografis berdekatan - Walantaka (106.2188) paling
# dekat, lalu Serang/Karangantu/Cilegon/Krenceng/Merak semuanya lebih barat
# lagi. Batasnya diverifikasi terhadap koordinat asli tiap nama tempat di
# atas, bukan ditaksir dari peta.
JABODETABEK_BBOX = {"south": -6.60, "north": -5.80, "west": 106.23, "east": 107.20}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "production-ml-predictive-maintenance/1.0 (internal tool)"
MIN_SECONDS_BETWEEN_REQUESTS = 1.1  # kebijakan Nominatim: maksimum 1 req/detik

_lock = threading.Lock()
_last_request_at = 0.0


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Cache geocoding rusak, mulai dari kosong: %s", CACHE_PATH)
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _within_jabodetabek(lat: float, lon: float) -> bool:
    box = JABODETABEK_BBOX
    return box["south"] <= lat <= box["north"] and box["west"] <= lon <= box["east"]


def _throttle() -> None:
    """Jaga jarak minimum antar-permintaan ke Nominatim."""
    global _last_request_at
    elapsed = time.time() - _last_request_at
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


def _query_nominatim(name: str) -> list[dict]:
    _throttle()
    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": f"{_search_query(name)}, Indonesia",
            "format": "json",
            "limit": 3,
            "countrycodes": "id",
            # Bias pencarian ke Jabodetabek tanpa memaksanya (bounded=0) -
            # penyaringan sesungguhnya tetap lewat _within_jabodetabek() di
            # bawah, supaya hasil yang jatuh persis di tepi tidak hilang.
            "viewbox": (
                f"{JABODETABEK_BBOX['west']},{JABODETABEK_BBOX['north']},"
                f"{JABODETABEK_BBOX['east']},{JABODETABEK_BBOX['south']}"
            ),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _looks_like_public_station(name: str) -> bool:
    """Nama ini pola stasiun kereta publik, atau fasilitas internal?

    Diperiksa terhadap SELURUH 153 lokasi di data: 142 berpola "STASIUN ..."
    dan 6 berakhiran "... (KA BANDARA)" - keduanya stasiun kereta publik yang
    datanya ada di OpenStreetMap. Sisanya 5 nama ("GUDANG NI", "SERVICE
    CENTER", "DIPO DEPOK", "IT KCI JUANDA", dan satu salah ketik "SRASIUN
    RAWA BUAYA") adalah fasilitas internal perusahaan atau typo - mencoba
    men-geocode nama seperti itu justru berisiko: dicoba, "SERVICE CENTER"
    ketemu, tapi ke gerai servis HP yang tidak terkait sama sekali, kebetulan
    berada di Jakarta juga sehingga lolos kotak Jabodetabek.

    Nama yang tidak lolos di sini TIDAK PERNAH dikirim ke Nominatim - bukan
    hanya disaring setelah hasil kembali - supaya tidak ada peluang kebetulan
    ketemu tempat yang salah.
    """
    upper = name.strip().upper()
    return upper.startswith("STASIUN ") or upper.endswith("(KA BANDARA)")


def _search_query(name: str) -> str:
    """Nama lokasi -> kalimat pencarian. "(KA BANDARA)" bukan bagian dari
    nama tempatnya, jadi dibuang; ditambahkan "Stasiun" di depan supaya
    Nominatim mencari stasiun, bukan sembarang tempat bernama sama."""
    upper = name.strip().upper()
    if upper.endswith("(KA BANDARA)"):
        base = name[: -len("(KA BANDARA)")].strip()
        return f"Stasiun {base}"
    return name


def _resolve_one(name: str) -> dict:
    """Geocode satu nama lokasi. Selalu mengembalikan entri cache yang valid,
    baik berhasil maupun tidak - supaya lokasi yang gagal tidak dicoba ulang
    setiap saat."""
    if not _looks_like_public_station(name):
        return {
            "resolved": False,
            "retry": False,
            "reason": "bukan nama stasiun publik (fasilitas internal atau typo)",
            "checked_at": time.time(),
        }

    try:
        results = _query_nominatim(name)
    except requests.RequestException as error:
        logger.warning("Geocoding gagal untuk %r: %s", name, error)
        # Tidak ditandai checked_at supaya dicoba lagi nanti - ini kegagalan
        # jaringan, bukan bukti bahwa lokasinya memang tidak bisa ditemukan.
        return {"resolved": False, "retry": True}

    for candidate in results:
        try:
            lat, lon = float(candidate["lat"]), float(candidate["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if _within_jabodetabek(lat, lon):
            return {
                "resolved": True,
                "lat": lat,
                "lon": lon,
                "matched_name": candidate.get("display_name"),
                "checked_at": time.time(),
            }

    # Ada hasil atau tidak, tidak ada yang lolos kotak Jabodetabek - lebih
    # baik dilaporkan "tidak ketemu" daripada memasang pin di tempat yang
    # mungkin salah.
    return {"resolved": False, "retry": False, "checked_at": time.time()}


def known_coordinates(locations: list[str]) -> dict[str, dict | None]:
    """Koordinat dari cache saja, TANPA memanggil jaringan.

    None berarti lokasi ini belum pernah dicoba sama sekali. `resolved: False`
    berarti sudah dicoba dan tidak lolos penyaringan.
    """
    cache = _load_cache()
    return {name: cache.get(name) for name in locations}


def resolve_missing(locations: list[str], budget_seconds: float) -> int:
    """Geocode lokasi yang belum ada di cache (atau perlu dicoba ulang),
    dibatasi anggaran waktu supaya satu request HTTP tidak menggantung lama.

    Mengembalikan jumlah lokasi yang berhasil diproses (baik ketemu maupun
    tidak) dalam anggaran waktu ini.
    """
    with _lock:
        cache = _load_cache()
        pending = [
            name for name in locations
            if name not in cache or cache[name].get("retry")
        ]
        if not pending:
            return 0

        started = time.time()
        processed = 0
        for name in pending:
            if time.time() - started >= budget_seconds:
                break
            cache[name] = _resolve_one(name)
            processed += 1
            _save_cache(cache)  # simpan tiap langkah - progres tidak hilang kalau terputus
        return processed
