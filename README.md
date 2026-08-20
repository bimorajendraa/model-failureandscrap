# production_ml

Versi minimal dan siap pakai dari pipeline prediksi PART: **training,
retraining, prediction**, serta lapisan serving berupa **REST API dan
dashboard**. Tidak ada notebook, EDA, profiling, visualisasi, ablation study,
atau eksperimen di sini - semua itu tetap di `db_om_preparation/` sebagai
referensi research.

Database **hanya dibaca**. Folder ini tidak pernah membuat, mengubah, atau
menghapus object apa pun di database, dan tidak bergantung pada schema
`analytics` hasil research.

## Dua model, dua pertanyaan berbeda

```
PART terpasang normal
        |
        v
  MODEL KERUSAKAN   "Kapan PART ini akan rusak?"        -> predict()
        |            probabilitas rusak 30/60/90/120 hari
        v
   PART RUSAK (dibongkar, masuk bengkel)
        |
        v
  MODEL SCRAP       "Kerusakan ini berakhir dibuang?"   -> predict_scrap()
        |            probabilitas tidak bisa diperbaiki
        v
   Dibuang  /  Diperbaiki & dipasang lagi
```

| | Model kerusakan | Model scrap |
|---|---|---|
| Pertanyaan | Kapan rusak | Kalau rusak, apakah dibuang |
| Latih | `train.py` | `train_scrap.py` |
| Prediksi | `predict()` | `predict_scrap()` |
| 1 baris data | 1 PART pada 1 titik waktu | 1 kejadian kerusakan |
| Data latih | 356.100 baris, 5.876 kerusakan | 1.407 kerusakan, 46 dibuang |
| Fitur | 21 | 7 |
| ROC-AUC uji | 0,821 | 0,762 |

Keduanya berdiri sendiri dan tidak saling menggantikan. Rantai pembacaan
database dipakai bersama, jadi definisi "kerusakan", "siklus pemasangan", dan
pembersihan datanya dijamin sama untuk keduanya.

---

## Cara pakai

### 1. Persiapan (sekali saja)

```bash
pip install -r requirements.txt
cp .env.example .env      # lalu isi kredensial database
```

### 2. Training / retraining

```bash
python train.py
```

Jalankan lagi kapan pun data di database sudah bertambah. Hasilnya tersimpan
sebagai versi baru (`models/v2/`, `models/v3/`, ...). Model production hanya
diganti kalau versi baru **tidak lebih buruk** pada data uji.

### 3. Prediction

```python
from predict import predict

predict("011201100101164")
```

```python
{
    "item_id": "011201100101164",
    "failure_probability_30d": 0.0494,
    "failure_probability_60d": 0.0964,
    "failure_probability_90d": 0.141,
    "failure_probability_120d": 0.1835,
    "risk_level": "LOW",
    "model_version": "v1",
    "as_of": "2026-08-03 11:07:22",
}
```

Bisa juga dari terminal: `python predict.py 011201100101164`

Pemanggil **cukup memberi ID PART**. Umur, jumlah kerusakan, jumlah corrective,
client, lifecycle, dan seluruh fitur lain dihitung sendiri dari database.

Kalau PART tidak dikenal atau sedang tidak terpasang, `predict()` melempar
`ItemNotScorable` dengan penjelasan alasannya.

---

### Prediksi risiko dibuang (scrap)

```python
from predict_scrap import predict_scrap, predict_death_risk

predict_scrap("011201100101164")
```

```python
{
    "item_id": "011201100101164",
    "scrap_probability": 0.0259,    # sudah dikalibrasi - perkiraan persen
    "scrap_risk_level": "LOW",
    "scrap_risk_basis": "dibandingkan kerusakan lain yang masuk bengkel",
    "item_type": "MOTOR",
    "item_type_known_to_model": True,
    "model_version": "v1",
    "as_of": "2026-08-03 11:07:22",
}
```

Latih ulang dengan `python train_scrap.py`.

`predict_death_risk(item_id)` menggabungkan keduanya menjadi risiko PART
benar-benar mati dalam 30 hari - baca peringatannya di bagian "Model scrap".

---

## Aplikasi serving: API + dashboard

Kedua model di atas dibungkus menjadi aplikasi predictive maintenance
sederhana. **Tidak ada satu pun perhitungan model yang dipindah atau ditulis
ulang di lapisan ini** - seluruh fitur dan probabilitas tetap dihitung
`feature_builder.py`, `predict.py`, dan `predict_scrap.py`.

### Arsitektur

```
PostgreSQL  (read-only)
     |
     v
data_reader.py  -> feature_builder.py / scrap_features.py   <- ML core, tidak diubah
     |
     v
predict.py / predict_scrap.py                                <- ML core, tidak diubah
     |
     v
inference/       model_loader, predictor, batch_predictor,   <- INDEPENDEN dari FastAPI
                 recommendation, explanation, history            (lihat "Struktur inferensi")
     |
     +---------------------+---------------------------+
     v                                                 v
api/routes/ + api/services/ (FastAPI)      scripts/run_pipeline.py, run_prediction.py
     |                                          (entry point manual, tanpa server)
     v
dashboard/       Streamlit
```

Aturannya satu arah: **dashboard tidak pernah menyentuh database maupun
memuat model**, dan **`inference/` tidak pernah bergantung pada `api/`** -
arah ketergantungannya searah (api -> inference), bukan sebaliknya, supaya
predictor bisa dipakai dari script atau test tanpa server FastAPI hidup. Semua
angka yang tampil di layar datang lewat HTTP dari API, sehingga ambang risiko,
aturan rekomendasi, dan kredensial database hanya ada di satu tempat.

Pemanggil cukup mengirim `item_id`. Fitur ML - umur pemasangan, riwayat
kerusakan, corrective maintenance, client, lokasi, kondisi armada - dibangun
sendiri oleh ML core dan **tidak boleh** dikirim dari luar.

### Pemasangan

```bash
pip install -r requirements-serving.txt   # sudah termasuk requirements.txt
cp .env.example .env                      # lalu isi kredensial database
```

`requirements.txt` sengaja dibiarkan hanya berisi kebutuhan training, supaya
mesin yang hanya melatih model tidak perlu memasang FastAPI dan Streamlit.

### Environment variable

Semuanya dibaca dari `.env` (yang sudah masuk `.gitignore`) atau environment.
Tidak ada kredensial yang boleh ditulis di dalam kode atau image.

| Variabel | Bawaan | Keterangan |
|---|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | - | wajib; sesi database dipaksa read-only |
| `DB_SSLMODE` | `prefer` | |
| `BATCH_CACHE_TTL_SECONDS` | `3600` | umur hasil batch scoring sebelum dihitung ulang |
| `WARMUP_BATCH_ON_STARTUP` | `false` | hitung batch saat start, bukan saat request pertama |
| `DEFAULT_RECOMMENDATION_LIMIT` | `50` | |
| `MAX_RECOMMENDATION_LIMIT` | `500` | batas atas `?limit=` |
| `DATA_FRESHNESS_TTL_SECONDS` | `60` | seberapa cepat data baru terlihat aplikasi |
| `CORS_ALLOW_ORIGINS` | kosong | origin frontend browser, dipisah koma |
| `GEOCODE_BUDGET_SECONDS_DEFAULT` | `60` | anggaran waktu geocoding lokasi per panggilan `/api/v1/locations/map` |
| `GEOCODE_BUDGET_SECONDS_MAX` | `90` | batas atas anggaran, apa pun yang diminta lewat query param |
| `API_BASE_URL` | `http://127.0.0.1:8000` | dipakai dashboard untuk menemukan API |

### Menjalankan API

```bash
uvicorn api.main:app --reload
```

Dokumentasi interaktif: <http://127.0.0.1:8000/docs>

### Endpoint

| Endpoint | Kegunaan |
|---|---|
| `GET /health` | status aplikasi, versi model, kesegaran cache batch. `?check_database=true` untuk ikut menguji koneksi database |
| `GET /api/v1/model` | versi, target, fitur, ambang risiko, dan metrik uji kedua model |
| `GET /api/v1/parts/{item_id}/failure` | peluang rusak 30/60/90/120 hari |
| `GET /api/v1/parts/{item_id}/scrap` | peluang tidak bisa diperbaiki **jika** rusak |
| `GET /api/v1/parts/{item_id}/assessment` | gabungan keduanya + rekomendasi + faktor risiko |
| `GET /api/v1/parts/{item_id}/history` | tanggal kerusakan dan lokasi yang pernah tercatat, dari event mentah |
| `GET /api/v1/recommendations` | daftar prioritas hasil batch scoring |
| `GET /api/v1/overview` | angka ringkas armada + daftar teratas |
| `GET /api/v1/filters` | nilai filter yang benar-benar ada di data |
| `GET /api/v1/locations/map` | sebaran risiko per lokasi + koordinat (kalau ada) |
| `GET /api/v1/monitoring/metrics` | metrik monitoring kedua model (offline dari training + live dari populasi aktif) |
| `GET /api/v1/monitoring/metrics/failure`, `/scrap` | metrik monitoring satu model saja |

```bash
curl http://127.0.0.1:8000/api/v1/parts/011201100101164/assessment
```

```json
{
  "item_id": "011201100101164",
  "status": "SCORED",
  "as_of": "2026-08-03 11:07:22",
  "failure": {
    "failure_probability_30d": 0.0494,
    "failure_probability_60d": 0.0964,
    "failure_probability_90d": 0.141,
    "failure_probability_120d": 0.1835,
    "risk_level": "LOW",
    "model_version": "v1"
  },
  "scrap": {
    "scrap_probability": 0.0259,
    "scrap_risk_level": "LOW",
    "item_type": "MOTOR",
    "model_version": "v1"
  },
  "death_probability_30d": 0.00128,
  "recommendation": {
    "priority": "LOW",
    "action": "MONITOR",
    "message": "Risiko kerusakan rendah. Cukup dipantau."
  },
  "model_version": {"failure": "v1", "scrap": "v1"}
}
```

Daftar prioritas:

```bash
curl "http://127.0.0.1:8000/api/v1/recommendations?risk=HIGH&limit=50"
```

Filter yang tersedia: `search` (cocok sebagian ID PART), `risk`, `priority`,
`item_type`, `client`, `location`, `replacement_candidates_only`, `limit`,
`offset`.

Dipakai dari browser (React/Vue dan sejenisnya) perlu `CORS_ALLOW_ORIGINS`
diisi lebih dulu; default-nya tertutup, dan origin harus disebut eksplisit
alih-alih dibuka untuk semua. Streamlit tidak memerlukannya karena
panggilannya server-ke-server.

### Tiga jawaban yang harus dibedakan

| Keadaan | Jawaban |
|---|---|
| PART tidak ada di database | `404` + `{"status": "NOT_FOUND"}` |
| PART ada tetapi tidak bisa diskor | `200` + `{"status": "NOT_SCORABLE", "reason": ...}` |
| Database tidak bisa dibaca | `503` + pesan umum |

PART yang tidak bisa diskor **bukan** kegagalan sistem: PART yang sedang
tidak terpasang memang tidak punya risiko kerusakan yang perlu diperkirakan.
Data yang tidak ada tidak pernah diganti nilai karangan hanya supaya prediksi
tetap keluar. Client tidak pernah menerima DSN, kredensial, SQL, atau stack
trace - detail lengkapnya hanya masuk log server.

### Alur satu prediksi

```
GET /api/v1/parts/{item_id}/assessment
        |
        v
prediction_service.get_part_assessment(item_id)
        |
        +--> predict.predict(item_id)            risiko kerusakan 30/60/90/120 hari
        +--> predict_scrap.predict_scrap(item_id)  risiko scrap (bersyarat)
        +--> recommendation_service.recommend(...) tindakan operasional
        +--> explanation.risk_factors(...)         faktor risiko dari fitur
```

### Logic rekomendasi

Recommendation engine (`inference/recommendation.py`) **tidak punya
ambang sendiri**. Masukannya hanya kelompok risiko yang sudah ditetapkan
model - ambang angkanya dibekukan saat training: risiko kerusakan dari
probabilitas 30-hari tetap (`FAILURE_HIGH_PROBABILITY_THRESHOLD`,
`FAILURE_MEDIUM_PROBABILITY_THRESHOLD`), risiko scrap dari kapasitas kerja
tim (`SCRAP_CAPACITY_PER_MONTH`) - keduanya di `config.py`. Isinya satu tabel
keputusan supaya seluruh aturan terlihat sekaligus dan mudah diganti.

| Risiko kerusakan | Risiko scrap | Prioritas | Tindakan |
|---|---|---|---|
| HIGH | HIGH | CRITICAL | `INSPECT_AND_PREPARE_REPLACEMENT` |
| HIGH | MEDIUM / LOW | HIGH | `PRIORITIZE_INSPECTION` |
| MEDIUM | HIGH | MEDIUM | `SCHEDULE_INSPECTION_AND_REVIEW_STOCK` |
| MEDIUM | MEDIUM / LOW | MEDIUM | `SCHEDULE_INSPECTION` |
| LOW | apa pun | LOW | `MONITOR` |

Risiko scrap **tidak pernah menaikkan prioritas sendirian**. Angkanya
bersyarat terhadap kerusakan, jadi PART yang kecil kemungkinannya rusak tidak
menjadi mendesak hanya karena seandainya rusak sulit diperbaiki - yang berubah
hanya perlu-tidaknya menyiapkan pengganti.

PART yang risiko kerusakannya MEDIUM/HIGH **dan** risiko scrap-nya HIGH
ditandai `replacement_candidate`. Itu bukan vonis bahwa PART akan dibuang,
melainkan penanda bahwa menyiapkan pengganti lebih awal masuk akal.

### Batch scoring

Dashboard bertanya "PART mana yang paling perlu diperhatikan", bukan "berapa
risiko PART X". Memanggil `predict()` belasan ribu kali berarti belasan ribu
kali query database, jadi `inference/batch_predictor.py` membaca seluruh data
sekali lalu menjalankan model pada semua baris sekaligus:

```
ambil siklus + event (2 query)
        |
        v
current_observations -> attach_history -> attach_fleet_snapshot
        |
        v
project_features per langkah 30 hari -> predict_proba seluruh baris
        |
        v
kelompok risiko -> rekomendasi -> urutkan menurut prioritas
```

Sekitar **16.900 PART aktif dalam ~35 detik**; hasilnya di-cache
(`BATCH_CACHE_TTL_SECONDS`), jadi permintaan filter dan paging berikutnya
dilayani dalam hitungan milidetik tanpa menghitung ulang.

**Kesetaraan single vs batch dijaga test.** Keduanya memanggil fungsi
`feature_builder` dan model yang sama, sehingga hasilnya wajib identik - bukan
sekadar mirip. `tests/test_parity.py` membandingkan probabilitas keempat
horizon, kelompok risiko, dan probabilitas scrap secara persis, dan memeriksa
bahwa jumlah PART aktif dan jumlah HIGH pada batch sama dengan yang dicatat
`train.py` di metadata saat model dilatih.

Satu-satunya bagian yang ditulis ulang untuk batch adalah penyusunan kolom
mentah scrap (`scrap_features.current_state()` hanya melayani satu PART per
panggilan); test membandingkannya kolom per kolom dengan fungsi aslinya.

### Struktur inferensi: `inference/`

`inference/` membungkus ML core (predict.py, predict_scrap.py, feature_builder.py)
menjadi hasil prediksi siap pakai - **tanpa bergantung pada FastAPI sama
sekali**. Bisa dipanggil dari mana pun: route API, test, atau script CLI.

| Modul | Tanggung jawab |
|---|---|
| `model_loader.py` | muat kedua model + metadata sekali per proses, validasi versi |
| `predictor.py` | prediksi SATU PART - failure, scrap, assessment gabungan, faktor risiko, riwayat |
| `batch_predictor.py` | prediksi SELURUH PART aktif sekaligus (vectorized) - lihat "Batch scoring" di atas |
| `recommendation.py` | terjemahkan kelompok risiko jadi tindakan operasional |
| `explanation.py` | faktor risiko dalam bahasa manusia, dari fitur yang benar-benar dipakai model |
| `history.py` | tanggal kerusakan/lokasi mentah untuk mendukung faktor risiko |

Single vs batch **sengaja tetap dua implementasi** (`predictor.py` vs
`batch_predictor.py`), bukan dipaksa jadi satu fungsi: batch memvektorkan
seluruh populasi dalam satu query, sementara single membaca riwayat satu PART
- memaksakan satu jalur kode berarti salah satu jadi sangat lambat (single
lewat loop batch) atau sangat lambat sebaliknya (batch memanggil predict()
per item, terbukti perlu puluhan ribu query). Kesetaraan **angkanya**
dijamin lewat `tests/test_parity.py`, bukan lewat berbagi kode baris-per-baris.

`api/services/` sekarang HANYA menyisakan yang murni urusan API (bukan
inferensi): `geocoding_service.py` (koordinat untuk peta) dan
`monitoring_service.py` (agregasi metrik untuk endpoint monitoring, memanggil
`inference.batch_predictor` seperti route lainnya).

### Menjalankan pipeline dan batch prediction manual

Dua entry point CLI, dipakai lewat `inference/` yang sama persis dengan API -
hasilnya tidak mungkin berbeda dari yang dilihat lewat HTTP:

```bash
python scripts/run_pipeline.py
```

Extract (`data_reader`) -> transform + feature build (`feature_builder`) untuk
seluruh PART aktif, tanpa memuat model dan tanpa prediksi - membuktikan
pipeline data berjalan berdiri sendiri. Tidak menyimpan apa pun.

```bash
python scripts/run_prediction.py                    # cetak 10 teratas
python scripts/run_prediction.py --top 20            # cetak 20 teratas
python scripts/run_prediction.py --output hasil.csv  # simpan semua ke CSV
```

Batch prediction manual lewat `inference.batch_predictor` - fungsi yang SAMA
dipakai `GET /api/v1/recommendations`. Belum menulis ke "prediction
database" (lihat bagian "Yang sengaja belum dikerjakan") - `--output` hanya
CSV lokal untuk pemeriksaan manual.

Keduanya memakai `logging` standar (bukan Grafana/Prometheus): `pipeline
started`, `database connected`, `rows extracted`, `features generated`,
`model loaded`, `prediction completed`, dan `error` kalau gagal.

### Menjalankan dashboard

API harus sudah jalan lebih dulu.

```bash
streamlit run dashboard/app.py
```

<http://localhost:8501> - empat halaman:

| Halaman | Isi |
|---|---|
| **Overview** | jumlah PART aktif, risiko tinggi/sedang, kandidat penggantian, daftar teratas |
| **Prioritas Perawatan** | seluruh PART aktif dengan filter risiko / jenis / client / lokasi |
| **Detail PART** | cari `item_id`: risiko 30/60/90/120 hari, risiko scrap, rekomendasi, faktor risiko, versi model |
| **Perencanaan Penggantian** | PART yang risiko rusak dan risiko scrap-nya sama-sama tinggi |
| **Peta Risiko** | sebaran lokasi PART aktif menurut risiko, dengan koordinat hasil geocoding |

Klik satu baris di tabel manapun untuk memilihnya, lalu tombol "Lihat detail"
muncul dan membawa ke halaman Detail PART - tanpa mengetik ulang ID.

Di halaman Detail PART, faktor risiko yang berupa hitungan ("2 kerusakan
tercatat dalam 365 hari terakhir") bisa dibuka lebih lanjut lewat dua tabel:
tanggal setiap kerusakan, dan riwayat lokasi (kapan pertama/terakhir tercatat
di tiap tempat) - keduanya dari event mentah, bukan hitungan ulang.

Tampilan selalu menyebut **peluang dalam jangka waktu** ("Rusak dalam 30 hari:
4,9%"), tidak pernah tanggal kerusakan - model memang tidak memperkirakan
tanggal.

### Faktor risiko (explanation)

Halaman detail menampilkan kondisi nyata PART yang menjadi masukan model -
riwayat kerusakan setahun terakhir, corrective maintenance, umur pemasangan,
kondisi armada model PART yang sama, rata-rata umur siklus sebelumnya,
perpindahan lokasi. Seluruhnya dibaca dari kolom yang benar-benar dihitung
`feature_builder.py`; tidak ada alasan yang dikarang.

Yang **tidak** diklaim: seberapa besar satu faktor menaikkan skor. Untuk itu
perlu analisis kontribusi per-fitur (SHAP), dan itu pekerjaan tahap
berikutnya. Setiap jawaban API membawa catatan ini apa adanya.

### Kesegaran data

Dua hal bisa membuat aplikasi diam-diam menyajikan angka lama, dan keduanya
ditutup oleh `inference/data_state.py`:

1. **Potret kondisi armada.** `predict.py` menyimpannya di variabel
   level-modul dan mengembalikannya tanpa memeriksa ulang batas waktu data -
   benar untuk proses CLI yang hidup beberapa detik, tetapi di server yang
   hidup berhari-hari membuat 3 fitur armada BEKU sementara 18 fitur lain ikut
   segar. Tidak ada error, prediksi tetap keluar, hanya angkanya yang keliru.
2. **Cache batch scoring**, yang semula hanya kedaluwarsa karena umur.

Batas waktu data diperiksa berkala (`DATA_FRESHNESS_TTL_SECONDS`). Begitu
terbukti bergeser, potret armada dibuang supaya ML core membangunnya ulang
dengan pemeriksaannya sendiri, dan hasil batch ditandai basi. `predict.py`
tidak diubah sama sekali - penutupannya dilakukan dari luar.

### Satu request, satu kali baca

Menilai satu PART memanggil `predict()` dan `predict_scrap()`, dan keduanya
membaca hal yang sama dengan argumen yang sama: batas waktu data, siklus, dan
event PART itu. Semula setiap panggilan membuka koneksi sendiri - **9 koneksi,
9 detik** untuk satu endpoint assessment.

`inference/query_cache.py` menyatukannya: selama satu request, pembacaan
dengan argumen yang sama dijawab dari hasil pertama. Hasilnya **3 koneksi, 4,8
detik**, dengan angka prediksi yang persis sama.

Cache-nya hanya hidup di dalam `request_scope()` dan hanya untuk thread itu -
di luar scope fungsi aslinya dipanggil apa adanya, jadi `train.py` dan
`python predict.py` dari terminal tidak terpengaruh sama sekali.

Halaman detail juga memakai fitur yang sudah dihitung batch kalau tersedia,
sehingga faktor risiko tidak perlu membaca database lagi. Kalau batch belum
pernah jalan, snapshot-nya dibangun untuk satu PART saja - menjelaskan satu
PART tidak pernah memaksa seluruh armada diskor.

### Peta lokasi: geocoding tanpa menebak koordinat

Database ini hanya menyimpan NAMA lokasi ("STASIUN JUANDA"), bukan koordinat
GPS. `api/services/geocoding_service.py` mencarinya lewat OpenStreetMap
Nominatim, tetapi disaring ketat karena geocoding polos terbukti berbahaya:
dicoba langsung pada 153 nama lokasi asli di data, "SERVICE CENTER" (nama
gudang servis internal) memang ketemu hasil - tapi nyangkut ke gerai retail
yang sama sekali tidak terkait, hanya karena kebetulan berada di area yang
sama. **Pin yang salah tempat lebih menyesatkan daripada tidak ada pin sama
sekali** untuk keputusan operasional.

Dua lapis penyaringan menutup celah itu:

1. **Sebelum dikirim ke Nominatim** - hanya nama berpola stasiun kereta
   publik ("STASIUN ..." atau "... (KA BANDARA)") yang dicoba. Dari 153
   lokasi di data, 148 berpola itu; sisanya (nama fasilitas internal seperti
   "GUDANG NI", "SERVICE CENTER", "DIPO DEPOK", atau salah ketik seperti
   "SRASIUN RAWA BUAYA") tidak pernah dikirim ke jaringan sama sekali -
   bukan sekadar hasilnya dibuang, supaya tidak ada peluang kebetulan ketemu
   tempat yang salah.
2. **Sesudah hasil kembali** - koordinatnya harus jatuh di dalam kotak
   Jabodetabek. Bukan angka yang dikarang: seluruh client yang tercatat di
   data (KCI, LRT Jabodebek, Railink bandara) beroperasi di situ, jadi
   kotak ini adalah batas yang didukung data itu sendiri.

Lokasi yang tidak lolos TIDAK ditampilkan sebagai pin - dilaporkan terpisah
di halaman Peta Risiko, tetap diurutkan menurut risiko, supaya PART berisiko
tinggi di lokasi itu tidak hilang dari pandangan hanya karena belum ada
koordinatnya.

Hasilnya di-cache di `.cache/geocode.json` (tidak masuk git, regenerable):
nama lokasi tidak berubah dari hari ke hari, jadi tidak digeocode ulang
setiap kali peta dibuka. Mematuhi kebijakan pemakaian Nominatim (User-Agent
deskriptif, maksimum 1 permintaan/detik); satu panggilan `/api/v1/locations/map`
dibatasi anggaran waktu (`GEOCODE_BUDGET_SECONDS_DEFAULT`) supaya tidak
menggantung lama - lokasi yang belum sempat dicoba diselesaikan pada
panggilan berikutnya, tombol "Coba cari koordinat lagi" di dashboard memicu
percobaan lanjutan itu.

### Promosi model: window evaluasi yang adil

`train.py` dan `train_scrap.py` tidak lagi membandingkan skor kandidat dengan
metrik LAMA yang tersimpan di metadata model production. Itu keliru: metrik
lama dihitung pada test split model itu SENDIRI saat ia dilatih, dan window
itu bergeser maju setiap tahun (`test_start` dihitung dari tahun `data_end`)
- kandidat dan model lama akhirnya dibandingkan pada dua periode yang
berbeda.

Sekarang model production (incumbent) dijalankan ULANG pada test split yang
PERSIS SAMA dengan kandidat (`evaluate_incumbent()`), termasuk memakai
dukungan tipe PART yang DIBEKUKAN miliknya sendiri - bukan dukungan baru
milik kandidat, supaya keduanya dievaluasi dengan metodologi fitur yang
identik, bukan hanya periode yang identik.

Promosi butuh **PR-AUC dan Recall@kapasitas-kerja** sama-sama tidak
memburuk - bukan ROC-AUC sendirian, yang bisa terlihat bagus walau presisi
pada kapasitas kerja nyata memburuk untuk data yang timpang begini. ROC-AUC
dan Brier tetap dihitung dan disimpan di `metadata.json["promotion_comparison"]`
untuk konteks/audit.

```bash
python train.py            # PR-AUC, ROC-AUC, Recall/Precision@kapasitas, Brier
                            # dicetak untuk kandidat vs model production,
                            # dihitung pada window yang sama persis
python train.py --force-promote   # pakai kandidat walau lebih buruk
```

### Versi model

Lapisan serving mengikuti mekanisme versi yang sudah ada - `models/failure/CURRENT`
dan `models/scrap/CURRENT` - dan tidak punya mekanisme sendiri. Model dimuat
**sekali saat aplikasi start**, lalu dipakai ulang; tidak ada model yang
dimuat per request. Versinya ikut di setiap jawaban (`model_version`), di
`/health`, dan di sidebar dashboard, supaya tidak ada angka yang terbaca tanpa
diketahui berasal dari model mana.

Untuk memakai model baru: latih (`python train.py`), lalu **restart** API.

### Training tetap terpisah dari inference

Tidak ada endpoint `POST /train` dan memang tidak akan ada. Training dijalankan
dari terminal, hasilnya dievaluasi, dan model baru hanya dipromosikan kalau
tidak lebih buruk pada data uji.

```
TRAINING                          INFERENCE
database                          database
   |                                 |
   v                                 v
train.py / train_scrap.py         model CURRENT
   |                                 |
   v                                 v
evaluasi -> models/vN -> CURRENT  FastAPI -> dashboard
```

### Production readiness

- **Logging terstruktur** (`api/logging_config.py`) - tanpa ini, log level
  INFO (model dimuat, batch scoring selesai, potret armada dibuang karena
  data bertambah) hilang diam-diam; Python hanya punya handler darurat untuk
  WARNING ke atas. Level diatur lewat `LOG_LEVEL` (bawaan `INFO`).
- **Connection pooling** (`api/db_pool.py`) - `data_reader.connect()`
  membuka satu koneksi baru per panggilan, benar untuk `predict.py`/`train.py`
  yang jadi proses CLI sekali pakai, tetapi boros untuk API yang melayani
  banyak request bersamaan. Ditambal transparan lewat monkeypatch saat API
  start (pola yang sama dengan `query_cache.py`) - `data_reader.py` sendiri
  tidak disentuh.
- **Dependency locking** (`requirements.lock.txt`) - snapshot versi PERSIS
  dari environment yang sudah diverifikasi 135 test lulus, untuk deployment
  yang butuh reproduksi environment yang sama persis. `requirements.txt` dan
  `requirements-serving.txt` tetap memakai rentang versi supaya fleksibel.

### Monitoring foundation

`GET /api/v1/monitoring/metrics` mengirim dua kelompok metrik yang SENGAJA
tidak dicampur:

- `offline` - hasil evaluasi SAAT TRAINING (PR-AUC, ROC-AUC, Precision/
  Recall@kapasitas, Brier), dibaca apa adanya dari `metadata.json` model yang
  sedang production. Berguna sebagai pengaman: kalau `CURRENT` tertukar ke
  model yang lebih buruk secara manual, angka ini langsung menunjukkannya.
- `live` - kondisi populasi PART aktif SEKARANG: sebaran skor, jumlah
  HIGH/MEDIUM dibandingkan dengan yang diharapkan dari training, pangsa PART
  dengan tipe yang tidak/kurang dikenal model (indikator awal butuh
  retraining), dan ringkasan fitur numerik utama.

PART yang sedang aktif belum punya label ground-truth (belum diketahui nanti
benar rusak atau tidak), jadi PR-AUC/ROC-AUC **live** secara matematis tidak
ada - itu sebabnya dua kelompok ini dipisah tegas, bukan digabung jadi satu
angka yang menyesatkan.

Ini fondasi, bukan sistem alert atau retraining otomatis - sengaja berhenti
di "menyediakan angka". Retraining otomatis baru masuk akal setelah
monitoring ini terbukti stabil.

### Test

```bash
pytest                       # seluruhnya - 135 test, ~6-7 menit, menyentuh database + internet
pytest tests/test_recommendation.py   # logic murni, tanpa database
```

Yang diuji: health, prediksi kerusakan/scrap satu PART, assessment gabungan,
PART tidak ditemukan (404), PART tidak bisa diskor, logic rekomendasi, batch
scoring, filter/pencarian/paging, CORS, tidak adanya endpoint training,
kesegaran data (potret armada basi dan cache batch), jumlah pembacaan
database per request, keempat halaman dashboard yang benar-benar dirender,
peta lokasi (penyaringan geografis + geocoding di-mock), promosi model
(window evaluasi adil, PR-AUC + Recall@kapasitas), konsistensi kolom fitur,
pemuatan/reload model, perlindungan kebocoran data masa depan, monitoring
foundation, serta - yang terpenting - **kesetaraan single vs batch**.

Test yang butuh database di-skip (bukan gagal) kalau database atau model tidak
tersedia. Di CI itu berbahaya: hasilnya terbaca "semua lulus" padahal tidak
ada yang diuji. Set `REQUIRE_DATABASE=1` supaya ketidaktersediaan menjadi
kegagalan.

### Yang sengaja belum dikerjakan (tahap ini)

Restrukturisasi `inference/` + `scripts/` menyiapkan batas yang jelas supaya
tahap local DB -> server DB berikutnya hanya perlu mengganti/menambah SUMBER
data (`config.db_settings()`), bukan membongkar ML code - tapi tahap itu
sendiri **belum** dikerjakan di sini, sesuai permintaan:

- Sinkronisasi/ingestion local database -> server database.
- Prediction database (hasil batch prediction disimpan permanen) - saat ini
  hanya cache in-memory (`batch_predictor`) dan CSV opsional lewat
  `--output`.
- Scheduler/cron untuk menjalankan pipeline otomatis - `scripts/run_pipeline.py`
  dan `scripts/run_prediction.py` murni manual.
- Airflow, Kafka, Spark, dbt, Redis, Celery, Kubernetes, MLflow Server,
  feature store, data warehouse, microservices, event streaming, automatic
  retraining, CI/CD kompleks, Grafana, Prometheus - tidak satu pun dibutuhkan
  untuk skala aplikasi ini sekarang.

### Docker

```bash
docker compose up --build
```

API di `localhost:8000`, dashboard di `localhost:8501`. Satu image dipakai
dua kali dengan perintah start berbeda - kodenya sama persis, jadi tidak ada
yang perlu dijaga sinkron. Database **tidak** ikut di-container: yang dipakai
adalah PostgreSQL yang sudah ada, kredensialnya dibaca dari `.env` di host dan
tidak pernah masuk ke dalam image.

---

## Struktur

```
production_ml/
├── config.py           # semua konstanta kedua model
├── data_reader.py      # SELECT read-only: event, siklus, kerusakan
├── feature_builder.py  # observasi + riwayat + armada + 21 fitur kerusakan
├── scrap_features.py   # label nasib kerusakan + 7 fitur model scrap
├── train.py            # latih model kerusakan
├── train_scrap.py      # latih model scrap
├── predict.py          # predict(item_id)
├── predict_scrap.py    # predict_scrap(item_id), predict_death_risk(item_id)
├── inference/           # PREDICTOR - independen dari FastAPI, bisa dipakai
│   │                    # dari script/test mana pun tanpa server hidup
│   ├── model_loader.py  # muat model + metadata sekali per proses
│   ├── predictor.py     # prediksi SATU PART
│   ├── batch_predictor.py  # prediksi SELURUH PART aktif (vectorized)
│   ├── recommendation.py   # terjemahkan risiko -> tindakan operasional
│   ├── explanation.py      # faktor risiko dalam bahasa manusia
│   └── history.py          # riwayat kerusakan/lokasi mentah
├── api/                # lapisan serving TIPIS - TIDAK menghitung model apa pun
│   ├── main.py         # aplikasi FastAPI + penanganan kesalahan
│   ├── schemas.py      # bentuk request/response
│   ├── settings.py     # pengaturan server (bukan konstanta model)
│   ├── errors.py       # tidak-ditemukan vs tidak-bisa-diskor
│   ├── logging_config.py  # setup logging terstruktur
│   ├── db_pool.py      # connection pooling database (server saja)
│   ├── query_cache.py  # dedup pembacaan dalam satu request HTTP
│   ├── data_state.py   # deteksi data bertambah, buang cache basi
│   ├── routes/         # health, model, prediction, recommendations, monitoring
│   └── services/       # HANYA urusan API: geocoding (peta), monitoring (agregasi)
├── dashboard/          # Streamlit; hanya bicara ke API lewat HTTP
│   ├── app.py          # Overview
│   └── pages/          # Prioritas, Detail PART, Perencanaan Penggantian, Peta
├── scripts/             # entry point manual - lihat "Menjalankan pipeline" di bawah
│   ├── run_pipeline.py   # extract -> transform -> feature build (tanpa prediksi)
│   └── run_prediction.py # batch prediction manual, lewat inference/ yang sama
├── tests/
├── Dockerfile
├── docker-compose.yml
├── models/
│   ├── failure/        # model KERUSAKAN
│   │   ├── CURRENT     # versi yang dipakai production
│   │   └── v1/         # model.cbm, calibrator.joblib, metadata.json
│   └── scrap/          # model SCRAP
│       ├── CURRENT
│       └── v1/         # model.joblib, metadata.json
├── requirements.txt          # kebutuhan training
├── requirements-serving.txt  # + FastAPI, Streamlit, test
└── README.md
```

Alur data, sama persis untuk training dan prediction:

```
database (tabel mentah)
      |
      v
data_reader      event operasional + siklus pemasangan
      |
      v
feature_builder  observasi -> riwayat + armada -> 21 fitur
      |
      +--------------------+
      v                    v
   train.py            predict.py
```

Fitur dihitung oleh **satu fungsi yang sama** (`feature_builder.build_features`)
untuk training maupun prediction, jadi tidak mungkin ada perbedaan antara fitur
yang dipelajari model dan fitur yang dipakai di production.

---

## Model

| | |
|---|---|
| Algoritma | CatBoost (200 iterasi, depth 4, lr 0.03, l2 10, kelas diseimbangkan) |
| Kalibrasi | Isotonic regression, dilatih pada data validasi |
| Target | PART mengalami kerusakan dalam 30 hari setelah tanggal observasi |
| Fitur | 21 (3 kategorikal + 15 riwayat PART + 3 kondisi armada) |
| Split | Berbasis waktu dengan jeda (embargo): tahun terakhir = uji, setahun sebelumnya = validasi |

Hasil training terakhir (data s/d 2026-08-03):

| Bagian | Baris | Kerusakan | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Latih | 251.568 | 3.852 | 0,8609 | 0,1335 |
| Validasi | 49.660 | 947 | 0,8167 | 0,1141 |
| Uji | 38.451 | 902 | **0,8211** | **0,1610** |

Brier terkalibrasi pada data uji: 0,0215. Lift PR-AUC: **6,86x** dibanding
menebak acak.

### 21 fitur final

Identitas & konteks: `part_model_category`, `client_category`
Umur: `installation_age_band`, `log_days_since_installation`
Riwayat: `log_total_prior_events`, `log_prior_failure_count`, `has_prior_failure`,
`log_prior_corrective_count`, `has_prior_corrective`,
`log_days_since_last_corrective`, `log_prior_distinct_places`
Jendela waktu: `log_prior_corrective_30d`, `log_prior_failure_365d`,
`log_prior_events_180d`
Lifecycle: `log_previous_cycle_lifetime_mean`, `has_previous_cycle`
Musiman: `month_sin`, `month_cos`
Kondisi armada: `model_failure_rate_90d`, `log_model_failures_90d`,
`log_model_fleet_size`

### Kondisi armada - fitur lintas-PART

Lima belas fitur pertama semuanya bicara tentang PART itu sendiri. Tiga fitur
terakhir melihat keadaan di sekelilingnya: **seberapa sering model PART ini
rusak dalam 90 hari terakhir**, dinormalkan per jumlah unit yang sedang
terpasang.

Bedanya dengan `part_model_category` penting: kategori hanya tahu *identitas*
model dan sifatnya statis, sedangkan laju armada tahu *kondisi terkini* -
menangkap cacat satu batch produksi, kohort yang menua bersama, atau masalah
musiman.

Dampaknya terukur (research: `db_om_preparation/reports/fleet_features_experiment.md`):

| | Tanpa armada | Dengan armada |
|---|---|---|
| ROC-AUC uji | 0,7947 | **0,8211** |
| PR-AUC uji | 0,1420 | **0,1610** |
| Tertangkap pada 200 PART teratas | 66 | **79** |

Selisih PR-AUC +0,0189 dengan 95% CI [+0,0129, +0,0255] - seluruhnya di atas
nol. `model_failure_rate_90d` menempati peringkat 2 dari 21 fitur.

**Konsekuensi pada prediksi**: fitur ini butuh riwayat kerusakan SELURUH model
PART, bukan hanya PART yang diminta. Membangunnya dari nol makan waktu ~45
detik, jadi potretnya **ikut disimpan bersama model** saat training dan dipakai
ulang selama data belum bertambah. `predict()` memeriksa `dataset_max_event_on`
lebih dulu; begitu ada kejadian baru, potretnya dihitung ulang supaya tidak
pernah memakai angka basi.

Hasilnya: panggilan pertama sekitar **7 detik**, berikutnya sekitar 4 detik.

### Risiko beberapa horizon

30/60/90/120 hari dihitung lewat **hazard chaining**: model 30 hari yang sama
dipakai berulang dengan fitur waktu dimajukan 30 hari tiap langkah, lalu peluang
bertahan dikalikan berantai. Keempat titik adalah kelipatan 30 hari, jadi
semuanya hasil chaining langsung tanpa interpolasi.

Cara ini menjamin `30d <= 60d <= 90d <= 120d` secara matematis. Pada pengujian
research, chaining mengalahkan classifier terpisah per horizon di ROC-AUC,
PR-AUC, maupun Brier.

Keterbatasan yang harus diingat: chaining mengasumsikan **tidak ada kejadian
baru** di antara langkah. Kalau PART benar-benar kena corrective bulan depan,
taksiran ini tidak "tahu" itu. Semakin jauh horizonnya, semakin besar pengaruh
asumsi tersebut.

### Kelompok risiko

Ambang tetap pada probabilitas kerusakan 30-hari yang sudah dikalibrasi -
angka yang sama persis dengan yang dibaca pengguna di layar, di `config.py`:

```python
FAILURE_HIGH_PROBABILITY_THRESHOLD = 0.25
FAILURE_MEDIUM_PROBABILITY_THRESHOLD = 0.15
```

**Ini bukan angka dari research** - keputusan operasional yang diambil setelah
memeriksa sebaran probabilitas seluruh armada aktif sungguhan (~16.900 PART).
Base rate kerusakan model ini rendah: PART paling berisiko sekalipun di
seluruh armada jarang melewati ~27% pada horizon 30 hari. Konsekuensinya
disadari sejak awal - jumlah PART yang ter-flag HIGH/MEDIUM jauh di bawah
kapasitas kerja tim (~200/bulan sebelumnya) dan bisa naik-turun signifikan
dari bulan ke bulan mengikuti kondisi armada, tidak lagi tetap sejumlah
kapasitas seperti sistem sebelumnya (lihat bagian "Kapasitas kerja tim" di
bawah untuk sistem lama, yang tetap dipakai mengevaluasi kualitas model saat
retrain, hanya bukan lagi dasar kelompok risiko).

### Kapasitas kerja tim (untuk evaluasi model, bukan kelompok risiko)

`FAILURE_CAPACITY_PER_MONTH` di `config.py` **tidak lagi** menentukan ambang
HIGH/MEDIUM di atas - sekarang hanya dipakai `training_utils.py` untuk
menghitung Recall/Precision@kapasitas saat membandingkan model kandidat
dengan model production ketika retrain (lihat `decide_promotion()` di
`train.py`). Angka ini mengukur "seandainya tim cuma bisa menindaklanjuti N
PART/bulan, seberapa banyak kerusakan sungguhan yang tertangkap" - metrik
kualitas model, terpisah dari kelompok risiko yang ditampilkan ke pengguna.

| Kapasitas/bln | Presisi | Tertangkap | Berapa kali lebih tepat |
|---|---|---|---|
| 50 | 29,4% | 145 dari 902 | 12,5x |
| 100 | 20,3% | 267 dari 902 | 8,6x |
| **200** (dipakai) | **16,6%** | **329 dari 902** | **7,1x** |
| 400 | 7,4% | 496 dari 902 | 3,2x |
| 800 | 7,4% | 633 dari 902 | 3,2x |

200/bulan dipilih karena **setara dengan aturan lama yang sudah tervalidasi di
research** (>=3x base rate validasi: presisi 16,6%, recall 36,6%).

---

## Model scrap

| | |
|---|---|
| Algoritma | Rata-rata Regresi Logistik + Random Forest (soft voting) |
| Target | Vonis `UNREPAIRABLE` atau `BROKEN` setelah kerusakan |
| Fitur | 7 |
| Periode | Mulai 2025-04-01, dengan embargo 30 hari di ujung data |
| Pemilihan model | PR-AUC rolling-origin pada 3 titik potong waktu |
| Ambang risiko | Dari kapasitas kerja bisnis, bukan default 0,5 |

Hasil pada data uji (323 kerusakan, 21 berakhir dibuang):

| Metrik | Nilai |
|---|---|
| ROC-AUC | 0,762 |
| PR-AUC | 0,255 (tebakan acak 0,065 - **naik 3,9x**) |
| Akurasi | 92,6% |
| Balanced accuracy | 67,2% |
| Presisi | 42,1% |
| Recall | 38,1% |

**Jangan pakai akurasi sebagai patokan.** Menebak "semua bisa diperbaiki"
menghasilkan akurasi 93,5% tetapi menangkap nol PART mati. Yang berarti di
sini ROC-AUC dan lift PR-AUC.

### Ambang risiko ditetapkan dari kapasitas kerja

Ambang **tidak** memakai default 0,5, dan **tidak** dioptimasi dari data uji.
Aturannya: model mengurutkan seluruh kerusakan menurut risiko, lalu sebanyak
kapasitas kerja per bulan itulah yang ditandai `HIGH`. Ambang dihitung dari
prediksi out-of-fold data latih, lalu hasilnya baru dilaporkan di data uji.

Satu angka yang mengatur semuanya, ada di `config.py`:

```python
SCRAP_CAPACITY_PER_MONTH = 3   # berapa PART/bulan yang sanggup ditindaklanjuti
```

Pilihan lain, diukur di data uji (~106 kerusakan masuk bengkel per bulan):

| Kapasitas/bln | Ambang | Presisi | Tertangkap |
|---|---|---|---|
| **3** (dipakai) | 0,68 | **42,1%** | 8 dari 21 |
| 5 | 0,64 | 30,8% | 8 dari 21 |
| 10 | 0,58 | 18,2% | 8 dari 21 |
| 15 | 0,52 | 16,7% | 10 dari 21 |
| 30 | 0,47 | 12,0% | 14 dari 21 |

Perhatikan baris 3 sampai 10: memperbesar daftar **tidak menambah tangkapan
sama sekali**, hanya menurunkan presisi. Jadi hanya ada dua titik yang masuk
akal - 3/bulan untuk daftar pendek yang tajam, atau 30/bulan kalau mengejar
tangkapan sebanyak mungkin. Ubah satu angka itu lalu jalankan ulang
`python train_scrap.py`.

Presisi 42,1% berarti hampir separuh PART yang ditandai memang benar-benar
dibuang - dibanding 6,5% kalau menebak acak, itu **6,5x lebih tepat sasaran**.

### 7 fitur

| Fitur | Arti |
|---|---|
| `log_age_total` | Umur PART sejak pertama kali tercatat |
| `log_cycle_age` | Sudah berapa lama terpasang sebelum rusak |
| `log_prior_repaired_count` | Berapa kali pernah berhasil diperbaiki |
| `has_prior_repair` | Pernah diperbaiki atau belum sama sekali |
| `log_prior_failure_count` | Jumlah kerusakan sebelumnya |
| `is_first_failure_ever` | Apakah ini kerusakan pertamanya |
| `item_type_category` | Jenis PART (yang riwayatnya sedikit digabung) |

Polanya: **PART tua yang baru pertama kali rusak dan belum pernah diperbaiki
cenderung langsung dibuang.** PART yang sudah pernah berhasil diperbaiki
terbukti masih bisa diperbaiki lagi.

### Angkanya sudah dikalibrasi, tetapi tetap perkiraan

`scrap_probability` boleh dibaca sebagai persentase - tetapi **perkiraan**,
bukan angka pasti.

Sebelum dikalibrasi, model mengeluarkan angka 0,3-0,7 padahal kenyataannya
hanya 3,3% kerusakan berakhir dibuang: meleset sekitar 12x. Platt scaling
(regresi logistik satu variabel) memperbaikinya:

| | Sebelum | Sesudah |
|---|---|---|
| Rata-rata keluaran | 41,2% | **2,5%** |
| Brier | 0,1851 | **0,0603** |
| ROC-AUC | 0,762 | 0,762 (tidak berubah) |

Sigmoid bersifat monoton, jadi **urutannya dijamin tidak berubah** - kalibrasi
hanya memperbaiki skalanya. Isotonic (yang dipakai model kerusakan) sengaja
TIDAK dipakai di sini: dengan kejadian sesedikit ini ia hanya menghasilkan 8
nilai berbeda dan justru merusak urutan (ROC turun ke 0,699).

**Catatan penting**: kalibrator dipasang pada data latih yang tingkat scrap-nya
2,3%, sementara belakangan naik ke 6,5%. Jadi angkanya cenderung
**merendahkan** risiko sesungguhnya. Urutannya tetap yang paling bisa
dipercaya.

### Kelompok risiko dibandingkan terhadap apa

`scrap_risk_level` membandingkan sebuah kerusakan dengan **kerusakan lain yang
masuk bengkel** - itu populasi tempat ambangnya dikalibrasi (kapasitas 3 per
bulan dari ~91 kerusakan per bulan).

Kalau `predict_scrap()` dipakai pada PART yang masih sehat ("seandainya rusak
besok"), kelompoknya tetap dibaca dengan dasar yang sama, dan hasilnya **45,4%
PART aktif masuk HIGH**. Itu bukan salah hitung: PART aktif memang didominasi
yang belum pernah rusak, dan itulah justru profil yang paling sering dibuang.
Tetapi jangan diperlakukan sebagai daftar pendek - untuk PART sehat, pakai
`predict_death_risk()` dan urutkan berdasarkan skornya.

### Cara label ditentukan

- **Dibuang**: vonis bengkel `UNREPAIRABLE` atau `BROKEN`.
- **Diperbaiki**: vonis `REPAIRED`, **atau** PART terbukti dipasang kembali.
- **Tidak dipakai**: tidak keduanya - bisa jadi dibuang tanpa dicatat, bisa
  jadi masih di bengkel, dan tidak ada cara membedakannya.

Memakai vonis bengkel saja akan membuang ratusan kerusakan yang sudah terbukti
selamat lewat pemasangan ulang, dan membuat model hanya belajar dari episode
yang kebetulan dicatat. Bias itu sempat membuat base rate terlihat meledak dari
4,3% ke 23,5% dalam satu kuartal; setelah pemasangan ulang ikut dihitung,
angkanya jadi 1,0% ke 6,2%.

**Embargo 30 hari** dipakai karena bukti datang dengan kecepatan berbeda: vonis
"dibuang" muncul median 2,9 hari, bukti "diperbaiki" lewat pemasangan ulang
butuh sampai 30 hari. Tanpa embargo, periode terbaru akan tampak penuh
kerusakan fatal semata-mata karena bukti selamatnya belum sempat muncul.

### Menggabungkan dua model

```
risiko MATI 30 hari = P(rusak dalam 30 hari) x P(dibuang | rusak)
```

Sudah dibacktest pada 74.412 observasi (37 benar-benar mati):

| Skor | ROC-AUC | Lift |
|---|---|---|
| Model kerusakan saja | 0,789 | 5,2x |
| Model scrap saja | 0,587 | 1,5x |
| **Gabungan** | **0,812** | **7,2x** |

Selisihnya nyata: 100% dari 500 resampling memihak gabungan.

**Tetapi kejadiannya sangat jarang** - sekitar 2-3 PART mati per bulan dari
belasan ribu PART aktif. Menandai 2.000 PART hanya menangkap 11 dari 37.
Jadi `predict_death_risk()` cocok sebagai **daftar pantau perencanaan stok**,
bukan pemicu tindakan per PART. Untuk keputusan per PART, pakai
`predict_scrap()` di saat kerusakan - di sana populasinya hanya ~96 kerusakan
per bulan dan modelnya jauh lebih tajam.

Backtest lengkap: `db_om_preparation/src/backtest_combined_death_risk.py`.

### Keterbatasan model scrap

1. **Kejadiannya sedikit** - 46 dibuang dari 1.407 kerusakan. Rentang
   ketidakpastian ROC-AUC lebar (95% CI kira-kira 0,67-0,85). Layak sebagai
   alat bantu prioritas, **belum layak jadi keputusan otomatis**.
2. **Kerusakan tanpa vonis dan tidak pernah dipasang lagi tidak bisa dilabeli**
   (873 kejadian). Kalau banyak di antaranya ternyata dibuang tanpa dicatat,
   model ini melihat gambaran yang terlalu optimis. Perbaikannya ada di
   disiplin pencatatan bengkel, bukan di model.
3. **Jenis PART yang belum dikenal otomatis dinilai berisiko tinggi**, karena
   masuk kelompok "jarang" yang kebetulan sering dibuang. Hasil prediksi
   menandainya lewat `item_type_known_to_model` - perlakukan yang `False`
   dengan hati-hati.
4. **Base rate masih naik antar-kuartal**, jadi tampilkan peringkat atau
   kelompok risiko, bukan angka persentase.

## Bukti kesetaraan dengan research

Rantai pembersihan data dibangun ulang dari tabel mentah, lalu **dibandingkan
baris-per-baris** dengan view `analytics` hasil research:

| Yang dibandingkan | Hasil |
|---|---|
| Observasi training + 18 fitur dasar + target | 356.100 baris, cocok semua, **0 selisih** |
| Snapshot PART aktif + 18 fitur dasar | 16.877 baris, cocok semua, **0 selisih** |
| Kerusakan yang bisa dilabeli (model scrap) | 1.407 baris, 46 dibuang - **sama persis** |
| Jumlah siklus pemasangan | 24.045 (sama) |
| Ukuran split latih/validasi/uji | 251.568 / 49.660 / 38.451 (sama) |
| Jumlah kerusakan per split | 3.852 / 947 / 902 (sama) |

Selisih kecil pada metrik (uji 0,7947 di sini vs 0,7883 di research) berasal
dari **urutan baris** yang berbeda saat masuk ke CatBoost, bukan dari perbedaan
data - datanya sudah dibuktikan identik di atas.

Perbandingan ini hanya alat verifikasi sekali jalan; production tidak
membutuhkan schema `analytics` untuk beroperasi.

---

## Yang diambil dari research, dan yang tidak

**Diambil** (yang benar-benar dibutuhkan model final):

- Normalisasi kode + kanonikalisasi client/lokasi ke data master, termasuk
  pencocokan fuzzy. Bukan kosmetik: 31% baris menulis nama client dengan typo
  (`KERETE` vs `KERETA`), dan tanpa tahap ini fitur `client_category` baris-baris
  itu jatuh ke `UNKNOWN`.
- Pembuangan event RECON administratif dan tanggal tidak valid.
- Dua dasar penentuan kerusakan: pembongkaran korektif, dan pembongkaran
  preventif yang ternyata berakhir rusak sebelum dipasang lagi.
- Siklus pemasangan per PART, beserta cara siklus itu berakhir.
- Aturan kelayakan label: sebuah observasi hanya dipakai kalau hasilnya
  benar-benar bisa dipastikan.
- Pengelompokan tipe PART yang riwayatnya sedikit (< 300 observasi).

**Tidak diambil** (terbukti tidak dipakai model final):

- Fitur lokasi/TERMINAL dan seluruh hierarchy PART-TERMINAL - belum terbukti
  cukup bernilai pada ablation study.
- Hitungan relocation, preventive, repair-process, dan window 30/90 hari yang
  tidak masuk 18 fitur.
- Model 90/180 hari terpisah - kalah dari hazard chaining.
- Cox PH, Random Survival Forest, XGBoost AFT - semua kalah dari model resmi.
- Seluruh view EDA, profiling, data quality report, dan kolom audit.

---

## Dua penyimpangan yang disengaja

Keduanya membuat production lebih konsisten, dan dicatat di sini supaya tidak
terlihat seperti kelalaian.

**1. Dukungan historis tipe PART dibekukan saat training.**
Research menghitung ulang angka ini dari data terbaru setiap kali scoring.
Production memakai angka yang tersimpan di `metadata.json`. Alasannya:
kategori yang dikenal model adalah kategori pada saat model dilatih. Kalau
sebuah tipe PART melewati ambang 300 di antara dua kali training, menghitung
ulang akan memunculkan kategori yang belum pernah dilihat model. Angka ini ikut
diperbarui otomatis setiap `train.py` dijalankan. Saat ini nilainya identik
dengan hasil hitung ulang research.

**2. Batas kelompok umur memakai definisi SQL yang membuat data training.**
Umur pemasangan bersifat pecahan (mis. 90,4 hari). Definisi SQL research
memakai `< 91`, sementara kode Python research memakai batas `<= 90` - keduanya
berbeda untuk umur di antara 90 dan 91 hari. Production mengikuti definisi SQL,
karena itulah yang dipakai saat model belajar.

---

## Asal-usul setiap konstanta

Hampir semua angka di bawah berasal dari hasil research yang sudah diuji;
pengecualiannya ditandai eksplisit di kolom Asal:

| Konstanta | Nilai | Asal |
|---|---|---|
| Hyperparameter CatBoost | depth 4, lr 0,03, l2 10 | pencarian hyperparameter, notebook 05 |
| Jumlah iterasi | 200 (tetap, bukan early stopping) | early stopping berbasis AUC bisa berhenti sangat prematur pada validasi yang positifnya sedikit |
| Horizon target | 30 hari | model resmi research |
| Ambang dukungan tipe PART | 300 observasi | rare-category ablation |
| Batas kelompok umur | 91/181/366/731/1461 hari | definisi fitur SQL research |
| Ambang risiko kerusakan (HIGH/MEDIUM) | 25% / 15% probabilitas 30-hari | **bukan dari research** - keputusan operasional, dipilih dari sebaran probabilitas armada aktif sungguhan |
| Ambang risiko scrap | 3x dan 1x base rate validasi | diuji pada data uji 2026 |
| Ambang fuzzy | skor >= 0,90 dan selisih >= 0,08 | aturan pencocokan research |
| Alias lokasi disetujui | GUDANG NUTECH -> GUDANG NI | sudah diverifikasi reviewer research |
| Singkatan | JKT -> JAKARTA | sudah diverifikasi reviewer research |

---

## Catatan operasional

- **Waktu jalan**: `train.py` sekitar 1-2 menit. `predict()` sekitar 2 detik
  untuk satu PART (pemanggilan berikutnya dalam proses yang sama lebih cepat
  karena model dan mapping teks sudah dimuat).
- **`as_of`** menunjukkan tanggal kejadian terbaru di database, bukan waktu
  sekarang. Kalau database berhenti terisi, nilai ini berhenti bergerak - itu
  sinyal yang berguna, bukan bug.
- **Data uji kecil**: kalau data uji punya kurang dari 30 kerusakan, `train.py`
  mencetak peringatan. Metrik pada sampel sekecil itu sangat berisik.
- **Membaca saja**: koneksi dibuka dengan `default_transaction_read_only=on`,
  jadi query yang mencoba menulis akan ditolak PostgreSQL - bukan sekadar janji
  di dokumentasi.
