"""Konfigurasi model kedua: risiko PART dibuang (scrap).

Model failure (lihat config/failure.py) menjawab "kapan PART akan rusak".
Model ini menjawab lanjutannya: "kalau sudah rusak, apakah PART itu masih
bisa diperbaiki". Keduanya terpisah dan tidak saling menggantikan.
"""

from __future__ import annotations

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
