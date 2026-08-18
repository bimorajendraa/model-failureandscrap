"""Konfigurasi tunggal untuk pipeline production.

Semua angka/ambang di sini berasal dari hasil research yang sudah terbukti di
repository lama (db_om_preparation). Tidak ada nilai baru yang dikarang: lihat
README.md bagian "Asal-usul setiap konstanta".
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / "models"
# Satu folder per model, masing-masing berisi CURRENT + v1, v2, ... supaya
# tidak ada dua "v1" yang artinya berbeda.
FAILURE_MODEL_DIR = MODEL_DIR / "failure"

# --- Fitur final model (18 fitur, urutan wajib sama seperti saat training) ----
CATEGORICAL_FEATURES = [
    "part_model_category",
    "client_category",
    "installation_age_band",
]
NUMERIC_FEATURES = [
    "log_days_since_installation",
    "log_total_prior_events",
    "log_prior_failure_count",
    "has_prior_failure",
    "log_prior_corrective_count",
    "has_prior_corrective",
    "log_days_since_last_corrective",
    "log_prior_distinct_places",
    "log_prior_corrective_30d",
    "log_prior_failure_365d",
    "log_prior_events_180d",
    "log_previous_cycle_lifetime_mean",
    "has_previous_cycle",
    "month_sin",
    "month_cos",
]
# --- Fitur kondisi armada (lintas-PART) --------------------------------------
#
# Ke-15 fitur di atas semuanya bicara tentang PART itu sendiri. Tiga fitur ini
# melihat keadaan di sekelilingnya: seberapa sering model PART ini rusak
# belakangan, dan berapa unit yang sedang terpasang.
#
# Bedanya dengan part_model_category penting: kategori hanya tahu IDENTITAS
# model dan sifatnya statis, sedangkan laju armada tahu KONDISI TERKINI -
# menangkap cacat satu batch produksi, kohort yang menua bersama, atau masalah
# musiman.
#
# Terbukti menambah daya tebak (research: reports/fleet_features_experiment.md):
# ROC-AUC 0,7947 -> 0,8211, lift 6,05 -> 6,86, dengan 95% CI selisih PR-AUC
# [+0,0129, +0,0255] - seluruhnya di atas nol. Pada kapasitas 200 PART/bulan,
# tertangkap 79 kerusakan dibanding 66 sebelumnya.
FLEET_FEATURES = [
    "log_model_failures_90d",
    "model_failure_rate_90d",
    "log_model_fleet_size",
]
# Jendela waktu untuk menghitung laju kerusakan armada.
FLEET_WINDOW_DAYS = 90

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + FLEET_FEATURES

# --- Target dan observasi ---------------------------------------------------
# Target: PART mengalami failure onset dalam 30 hari SETELAH observation_on.
TARGET_HORIZON_DAYS = 30
# Snapshot training dibuat pada grid tetap 30 hari sejak tanggal pemasangan.
OBSERVATION_STEP_DAYS = 30

# --- Split waktu (dengan embargo selebar horizon target) --------------------
# Batas latih/validasi/uji TIDAK ditulis sebagai tanggal tetap: assign_split()
# di train.py menghitungnya dari tahun terakhir yang ada di data, supaya
# training ulang tahun depan tetap menguji pada periode terbaru.
MIN_OBSERVATION_DATE = "2014-01-01"

# --- Hyperparameter model ---------------------------------------------------
# Iterasi TETAP (bukan early stopping): early stopping berbasis AUC pada
# validasi yang positifnya sedikit terbukti bisa berhenti sangat prematur dan
# menghasilkan model dengan resolusi probabilitas sangat kasar.
CATBOOST_PARAMS = {
    "iterations": 200,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "use_best_model": False,
    "verbose": False,
    "thread_count": 1,
}
RANDOM_STATE = 42

# --- Konstanta feature engineering ------------------------------------------
# Tipe PART dengan riwayat < 300 observasi dikelompokkan jadi satu kategori
# supaya model tidak menghafal pola dari sampel yang terlalu kecil.
MIN_PART_MODEL_SUPPORT = 300
LOW_SUPPORT_LABEL = "LOW_HISTORICAL_SUPPORT"
UNKNOWN_LABEL = "UNKNOWN"

# Ambang batas umur (hari). Umur bersifat pecahan, jadi ambang ditulis sebagai
# batas "lebih kecil dari" persis seperti definisi SQL yang membuat data
# training: <91, <181, <366, <731, <1461.
AGE_BAND_THRESHOLDS = [91, 181, 366, 731, 1461]
AGE_BAND_LABELS = [
    "000_090_DAYS",
    "091_180_DAYS",
    "181_365_DAYS",
    "366_730_DAYS",
    "731_1460_DAYS",
    "1461_PLUS_DAYS",
]

# --- Prediksi ---------------------------------------------------------------
# Semua horizon adalah kelipatan 30 hari supaya setiap titik merupakan hasil
# hazard chaining langsung, tanpa interpolasi.
PREDICTION_HORIZON_DAYS = [30, 60, 90, 120]

# --- Toleransi bisnis model kerusakan ---------------------------------------
#
# Berapa PART per bulan yang sanggup diprioritaskan (disiapkan penggantinya,
# dijadwalkan pemeriksaan, atau ditaruh cadangannya di dekat lokasi). Model
# mengurutkan seluruh PART aktif menurut risiko, lalu sebanyak kapasitas
# inilah yang ditandai HIGH.
#
# Diukur pada data uji 2026 (sekitar 5.500 pemeriksaan PART per bulan):
#
#   kapasitas/bln  ambang   presisi   tertangkap    berapa kali lebih tepat
#              50  0,1365    29,4%    145 dari 902           12,5x
#             100  0,0994    20,3%    267 dari 902            8,6x
#             200  0,0882    16,6%    329 dari 902            7,1x
#             400  0,0450     7,4%    496 dari 902            3,2x
#             800  0,0372     7,4%    633 dari 902            3,2x
#
# Default 200/bulan dipilih karena SETARA dengan aturan lama yang sudah
# tervalidasi di research (>=3x base rate validasi: presisi 16,6%, recall
# 36,6%). Jadi perilakunya tidak berubah - yang berubah hanya cara
# menyetelnya, dari kelipatan statistik menjadi angka kapasitas yang bisa
# dibicarakan dengan tim operasional.
#
# Ubah SATU angka ini kalau kapasitas berubah, lalu jalankan `python train.py`.
FAILURE_CAPACITY_PER_MONTH = 200
# Kelompok MEDIUM = lapis berikutnya kalau kapasitas bertambah.
FAILURE_MEDIUM_CAPACITY_MULTIPLIER = 2.0

# --- Kanonikalisasi teks (client/lokasi) ------------------------------------
# Mapping yang sudah disetujui reviewer pada fase research. Disimpan sebagai
# konstanta supaya production tidak bergantung pada tabel di schema analytics.
APPROVED_LOCATION_ALIAS = {"GUDANG NUTECH": "GUDANG NI"}
APPROVED_CLIENT_ALIAS: dict[str, str] = {}
TEXT_ABBREVIATION_MAPPING = {"JKT": "JAKARTA"}

# Kandidat fuzzy diterima otomatis hanya kalau sangat mirip DAN jauh lebih
# mirip dibanding kandidat kedua.
FUZZY_MIN_SCORE = 0.90
FUZZY_MIN_MARGIN = 0.08


# ---------------------------------------------------------------------------
# Model kedua: risiko PART dibuang (scrap)
#
# Model di atas menjawab "kapan PART akan rusak". Model ini menjawab lanjutannya:
# "kalau sudah rusak, apakah PART itu masih bisa diperbaiki". Keduanya terpisah
# dan tidak saling menggantikan.
# ---------------------------------------------------------------------------

SCRAP_MODEL_DIR = MODEL_DIR / "scrap"

# Vonis bengkel yang berarti PART tidak kembali ke layanan.
SCRAP_STATUS = ("UNREPAIRABLE", "BROKEN")
# SENDLOG (BROKEN) ikut menutup episode walaupun bukan vonis akhir.
FAILURE_OUTCOME_STATUS = ("UNREPAIRABLE", "BROKEN", "SENDLOG (BROKEN)")
REPAIR_COMPLETED_STATUS = "REPAIRED"

# Status UNREPAIRABLE baru dipakai sejak 2025-04-23 bersama proses repair
# detail. Sebelum itu PART yang dibuang tidak bisa dibedakan dari PART yang
# sekadar hilang dari catatan, jadi tidak boleh ikut dilatih.
SCRAP_ERA_START = "2025-04-01"
# Bukti "dibuang" muncul cepat (median 2,9 hari), bukti "diperbaiki" lewat
# pemasangan ulang jauh lebih lambat (p80 = 30 hari). Tanpa embargo, periode
# terbaru akan tampak penuh kerusakan fatal semata-mata karena bukti selamatnya
# belum sempat muncul.
SCRAP_EMBARGO_DAYS = 30

# Jenis PART dengan episode lebih sedikit dari ini digabung jadi satu kategori.
SCRAP_MIN_TYPE_SUPPORT = 20

SCRAP_CATEGORICAL_FEATURES = ["item_type_category"]
SCRAP_NUMERIC_FEATURES = [
    "log_age_total",
    "log_cycle_age",
    "log_prior_repaired_count",
    "has_prior_repair",
    "log_prior_failure_count",
    "is_first_failure_ever",
]
SCRAP_FEATURE_COLUMNS = SCRAP_CATEGORICAL_FEATURES + SCRAP_NUMERIC_FEATURES

# Sengaja hanya 7 fitur: kejadian scrap sedikit, dan menambah fitur terbukti
# menurunkan performa sesungguhnya walaupun angka validasinya naik.
SCRAP_RANDOM_STATE = 42

# --- Toleransi bisnis: berapa PART per bulan yang sanggup ditindaklanjuti ----
#
# Ini ANGKA KEPUTUSAN BISNIS, bukan hasil hitungan statistik. Ambang risiko
# diturunkan dari sini: model mengurutkan seluruh kerusakan, lalu sebanyak
# kapasitas inilah yang ditandai HIGH.
#
# Kenapa kapasitas, bukan balanced accuracy: balanced accuracy diam-diam
# menganggap satu scrap yang kelewat sama ruginya dengan satu salah alarm.
# Di lapangan tidak begitu, dan yang benar-benar membatasi adalah berapa
# banyak PART yang sanggup disiapkan penggantinya lebih awal.
#
# Diukur pada data uji 2026 (sekitar 106 kerusakan masuk bengkel per bulan):
#
#   kapasitas/bln  ambang   presisi   tertangkap
#               3    0,68     42,1%     8 dari 21
#               5    0,64     30,8%     8 dari 21
#              10    0,58     18,2%     8 dari 21
#              15    0,52     16,7%    10 dari 21
#              30    0,47     12,0%    14 dari 21
#
# Perhatikan baris 3 sampai 10: memperbesar daftar TIDAK menambah tangkapan
# sama sekali (tetap 8 dari 21), hanya menurunkan presisi. Jadi hanya ada dua
# titik yang masuk akal - 3/bulan untuk daftar pendek yang tajam, atau
# 30/bulan kalau memang mengejar tangkapan sebanyak mungkin.
#
# Default 3/bulan: daftarnya pendek, hampir separuhnya benar-benar dibuang
# (42,1% vs 6,5% kalau menebak acak), dan realistis dikerjakan.
#
# Ubah SATU angka di bawah ini kalau kapasitas tim berubah, lalu jalankan
# ulang `python train_scrap.py`. Tidak ada yang lain yang perlu disentuh.
SCRAP_CAPACITY_PER_MONTH = 3
# Kelompok MEDIUM = lapis berikutnya yang akan dikerjakan seandainya kapasitas
# bertambah, dipakai sebagai daftar cadangan.
SCRAP_MEDIUM_CAPACITY_MULTIPLIER = 2.0
SCRAP_TEST_START = "2026-04-01"
# Titik potong untuk MEMERIKSA (bukan memilih) model. Semuanya wajib lebih
# awal dari SCRAP_TEST_START - kalau ada yang menyentuh periode uji, angka
# akhirnya tidak lagi jujur.
SCRAP_ROLLING_CUTOFFS = ["2025-10-01", "2026-01-01"]

# Model DITETAPKAN DI MUKA, tidak dipilih dari data.
#
# Alasannya diukur, bukan diasumsikan: fold pemeriksaan hanya berisi 7 dan 2
# kejadian "dibuang". PR-AUC pada sampel sekecil itu nyaris acak, sehingga
# "memilih model terbaik" darinya sama saja memilih dari derau - terbukti
# saat dicoba, pemenang fold justru yang paling buruk di data uji.
#
# Yang dipakai adalah rata-rata regresi logistik dan random forest. Dasarnya
# prinsip, bukan angka: keduanya salah dengan cara berbeda (yang satu
# menangkap kecenderungan lurus, yang lain ambang dan kombinasi), dan
# merata-ratakan dua model yang salahnya tidak searah menurunkan ragam
# tanpa perlu bukti dari sampel kecil.
#
# Tabel perbandingan tetap dicetak train_scrap.py sebagai PEMERIKSAAN -
# supaya kalau suatu saat ada kandidat yang unggul jauh melampaui derau,
# hal itu terlihat.
SCRAP_MODEL_NAME = "Gabungan LogReg + RF"


def db_settings() -> dict[str, str]:
    """Kredensial database dari .env / environment. Production hanya membaca."""
    load_dotenv(PACKAGE_DIR / ".env")
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
