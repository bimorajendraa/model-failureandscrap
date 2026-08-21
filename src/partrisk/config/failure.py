"""Konfigurasi model failure: fitur, hyperparameter, dan ambang operasional.

Hampir semua angka/ambang di sini berasal dari hasil research yang sudah
terbukti di repository lama (db_om_preparation) - lihat README.md bagian
"Asal-usul setiap konstanta". Pengecualian: FAILURE_HIGH/MEDIUM_PROBABILITY_
THRESHOLD adalah keputusan operasional yang dipilih belakangan berdasarkan
sebaran probabilitas armada aktif sungguhan, bukan dari research - lihat
komentar di masing-masing konstanta.
"""

from __future__ import annotations

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

# --- Kelompok risiko kerusakan (HIGH/MEDIUM/LOW) -----------------------------
#
# Ambang tetap pada probabilitas kerusakan 30-hari YANG SUDAH DIKALIBRASI -
# angka yang sama persis dengan yang dibaca pengguna di layar (bukan skor
# mentah). BUKAN dari research - keputusan operasional yang diambil setelah
# memeriksa sebaran probabilitas armada aktif sungguhan (~16.900 PART):
# PART paling berisiko sekalipun jarang melewati ~27% pada horizon 30 hari,
# jadi ambang 25%/15% ini dipilih sadar akan konsekuensinya - jumlah PART
# yang ter-flag HIGH/MEDIUM akan JAUH di bawah kapasitas kerja tim
# (~200/bulan) dan bisa naik-turun signifikan dari bulan ke bulan mengikuti
# kondisi armada, tidak lagi tetap sejumlah kapasitas seperti sistem lama.
#
# Ubah SATU angka ini kalau ambangnya perlu digeser, lalu jalankan
# `python train.py`.
FAILURE_HIGH_PROBABILITY_THRESHOLD = 0.25
FAILURE_MEDIUM_PROBABILITY_THRESHOLD = 0.15

# --- Kapasitas kerja tim (dipakai mengevaluasi kualitas model saat promosi) --
#
# BUKAN dasar kelompok risiko di atas (itu sudah ambang tetap) - ini dipakai
# training_utils.py untuk menghitung Recall/Precision@kapasitas, metrik yang
# membandingkan model kandidat vs model production saat retrain (lihat
# decide_promotion() di train.py). Berapa PART per bulan yang sanggup
# diprioritaskan tim, diukur pada data uji 2026 (sekitar 5.500 pemeriksaan
# PART per bulan):
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
# 36,6%).
FAILURE_CAPACITY_PER_MONTH = 200
