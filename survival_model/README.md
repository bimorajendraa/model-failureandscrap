# survival_model — eksperimen time-to-failure PART

Eksperimen survival analysis yang **sepenuhnya terpisah** dari model
production di root repo (`train.py`, `feature_builder.py`, dst.). Tidak ada
satu file pun di luar `survival_model/` yang diubah untuk membuat ini. Model
production tetap menjadi baseline; hasil di sini adalah **challenger
comparison**, bukan pengganti - lihat "Rekomendasi akhir" di bawah untuk
keputusan apa adanya.

## Objective

> Berapa lama sebuah PART dapat bertahan sebelum mengalami failure?

Model production (`train.py` di root) menjawab pertanyaan yang berbeda:
"apakah PART ini akan rusak dalam 30 hari ke depan?", lewat classification
snapshot pada grid 30-harian. Eksperimen ini memakai **survival analysis /
time-to-event**: satu baris = satu lifecycle PART, targetnya durasi sampai
failure (atau sampai disensor).

## Unit data: satu lifecycle/episode

Sumbernya `data_reader.get_cycles()` (fungsi READ-ONLY yang sudah ada di
root, di-reuse apa adanya) - satu baris per **installation cycle** (siklus
pemasangan), lengkap dengan cara siklus itu berakhir:

```
INSTALL
   |
   v
PART aktif
   |
   +--> FAILURE                              -> event=1
   +--> masih aktif sampai batas observasi    -> event=0 (censored)
   +--> REINSTALL tanpa failure tercatat      -> lihat "Censoring" di bawah
```

Kalau sebuah PART rusak, diperbaiki, lalu dipasang lagi, `get_cycles()`
sudah memberi `installation_cycle_id` yang berbeda untuk tiap pemasangan -
setiap siklus diperlakukan sebagai lifecycle terpisah (cycle 1 bisa
`event=1`, cycle 2 `event=1` lagi, cycle 3 `event=0`/censored), **bukan**
satu PART yang diulang jadi banyak baris snapshot bulanan.

Filter cohort: `is_initial_model_cohort` (identitas tipe PART cocok dengan
inventory - filter yang SAMA dipakai model classification) dan
`installed_on < cycle_end_on` (durasi positif).

## Target: `duration_days` dan `event_observed`

- `event_observed = 1`, `duration_days = failure_onset_on - installed_on` -
  PART benar-benar rusak.
- `event_observed = 0`, `duration_days = cutoff - installed_on` - PART
  **right-censored**: pada titik cutoff belum ada failure tercatat. Ini
  **bukan** "PART ini tidak akan pernah rusak" - hanya "belum rusak sampai
  titik ini yang kita tahu".

## Censoring: kenapa cutoff-nya per-split, bukan satu tanggal global

Draft awal eksperimen ini memakai SATU cutoff sensor global
(`dataset_max_event_on`, tanggal event terbaru di database) untuk semua
baris. Itu ternyata sumber leakage: lifecycle TRAIN yang dipasang bertahun-
tahun lalu dan masih aktif akan "tahu" ia bertahan sampai HARI INI, artinya
labelnya diam-diam membawa informasi tentang apa yang terjadi sepanjang
periode VALIDATION/TEST - versi survival dari alasan model classification
butuh embargo (lihat bagian Split di bawah), bedanya di sini window-nya
tidak terbatas 30 hari.

Perbaikannya: **administrative censoring per-split**. Tiap lifecycle diberi
`cutoff` sesuai split tempat `installed_on`-nya jatuh (`validation_start`
untuk TRAIN, `test_start` untuk VALIDATION, `data_end`/hari-ini untuk TEST -
lihat "Split" di bawah untuk definisi batasnya), lalu duration/event dihitung
ULANG terhadap cutoff itu, bukan terhadap nasib akhir lifecycle yang
sebenarnya:

```
kalau failure_onset_on ADA dan <= cutoff:
    event=1, duration = failure_onset_on - installed_on   # fakta historis, selalu valid
elif cycle_end_on > cutoff:
    event=0, duration = cutoff - installed_on             # bukti langsung: masih berjalan
                                                            # tepat di cutoff, apa pun nasib
                                                            # akhirnya nanti
else:  # cycle sudah berakhir pada/sebelum cutoff, tanpa failure -
       # hanya mungkin saat cutoff=data_end (TEST)
    kalau RIGHT_CENSORED_AT_DATA_END dan is_recon_verified_negative_eligible:
        event=0, duration = cycle_end_on - installed_on
    selain itu (REINSTALL_WITHOUT_RECORDED_FAILURE, atau RECON tak terverifikasi):
        EXCLUDE - status pada cutoff itu sendiri tidak bisa dipastikan
```

`is_recon_verified_negative_eligible` dan status `REINSTALL_WITHOUT_RECORDED_
FAILURE` datang langsung dari `get_cycles()` - definisi yang SAMA dipakai
model classification untuk membedakan "PART ini benar-benar terbukti belum
rusak" dari "cara siklus ini berakhir ambigu" (misalnya PART yang dipasang
ulang tanpa ada failure tercatat di antaranya, atau ada aktivitas RECON
administratif setelah kejadian operasional terakhirnya). Aturan itu **tidak
diciptakan ulang** di sini, hanya diterapkan pada cutoff yang berbeda-beda.

Efek sampingnya: rule per-split ini **memulihkan** kembali sebagian besar
lifecycle yang tadinya harus di-exclude sepenuhnya (REINSTALL, atau
RIGHT_CENSORED yang RECON-nya tak terverifikasi) - begitu di-truncate ke
cutoff split-nya sendiri (lebih awal dari kejadian ambigu itu), ambiguitasnya
jadi tidak relevan. Hasilnya: dari 23.927 lifecycle cohort, **20.116 eligible**
(84,1%) - jauh lebih hemat data dan lebih benar metodologinya dibanding
exclude blanket. Detail lengkap ada di `reports/data_validation.md`
(dihasilkan `build_dataset.py`).

## Leakage prevention

- **Fitur point-in-time**: seluruh fitur (riwayat, kondisi armada, musiman)
  dihitung pada `observation_on = installed_on` lewat
  `feature_builder.attach_history()`/`attach_fleet()` (fungsi asli, TIDAK
  diubah) - keduanya hanya memakai event `<= observation_on`, jadi tidak
  mungkin memakai informasi masa depan.
- **Tanpa embargo bergaya classification**: model classification butuh
  embargo (buang baris yang window 30-harinya menembus batas split) karena
  window resolusinya TETAP. Survival tidak punya window tetap (durasi bisa
  bertahun-tahun) - solusinya administrative censoring per-split di atas,
  bukan exclude berbasis embargo.
- **Leakage yang TETAP ada dan didokumentasikan, bukan diperbaiki** (sesuai
  prinsip jangan overengineering): PART yang punya banyak lifecycle bisa saja
  punya cycle 1 di TRAIN dan cycle 3 di TEST (fitur
  `previous_cycle_lifetime_mean`/`has_previous_cycle` menghubungkan
  keduanya). Ini BUKAN leakage temporal (urutan waktu tetap terjaga - cycle 1
  selalu terjadi sebelum cycle 3), tapi tetap potensi model "mengenali"
  identitas item lintas split. Pada data saat ini: **1.234 item (7,5% dari
  item unik)** punya lifecycle di lebih dari satu split - lihat
  `reports/data_validation.md`.

## Keterbatasan: baseline instalasi vs kondisi PART sekarang

**Ini keterbatasan yang paling penting untuk dipahami sebelum memakai model
ini.**

Fitur model dihitung PERSIS pada `installed_on` - kondisi PART saat siklus
ini dimulai. Model tidak pernah melihat apa yang terjadi SETELAH instalasi,
kecuali lewat berlalunya waktu pada kurva `S(t)` itu sendiri. Ini beda
mendasar dari model classification, yang fiturnya di-refresh tiap 30 hari
dan karena itu "tahu" kejadian terbaru (corrective/relokasi baru-baru ini).

Konsekuensinya untuk PART aktif yang sudah lama terpasang: proyeksi risiko
ke depan (`P(fail<=N hari | selamat sampai umur A) = 1 - S(A+N)/S(A)`) hanya
memperhitungkan **berlalunya waktu** sejak instalasi, BUKAN kejadian apa pun
yang terjadi selama PART itu berjalan. `predict.py` selalu menampilkan
`installed_on` DAN `as_of` berdampingan supaya ini terlihat jelas, bukan
tersirat.

Ini keterbatasan yang nyata (bukan cacat implementasi), konsisten dengan
"unit data = lifecycle/episode" yang jadi tujuan eksperimen ini - solusi
time-varying-covariate penuh (landmarking) sengaja tidak dikerjakan karena
akan mengembalikan bentuk grid 30-harian yang justru ingin dihindari (lihat
bagian berikutnya).

## Kenapa grid 30-harian classification TIDAK dipakai untuk melatih

Imbalance ekstrem pada model classification (356.100 baris, 2,3% positif)
berasal justru dari grid itu: satu PART yang sama diulang jadi baris negatif
setiap 30 hari selama ia belum rusak. Survival model di sini HANYA dilatih
dari 20.116 lifecycle di atas (satu baris per lifecycle) - event rate jauh
lebih sehat karena tidak diulang-ulang.

Grid 30-harian classification muncul lagi HANYA di evaluasi Lapis 2 (lihat
"Evaluasi" di bawah), dan HANYA dipinjam **read-only** (lewat
`train.build_dataset()` di root) sebagai populasi + label pembanding untuk
menilai skor survival model pada baris yang persis sama dengan yang dipakai
menilai model classification. Tidak ada fitting/training pada grid itu.

## Split temporal

Berdasar `installed_on` (bukan `observation_on`, karena unitnya sudah
lifecycle-level), memakai formula tahun yang SAMA dengan `train.assign_split`
di root: tahun terakhir data = TEST, setahun sebelumnya = VALIDATION,
`config.MIN_OBSERVATION_DATE` (2014-01-01) sebagai batas bawah.

| Split | installed_on | cutoff sensor |
|---|---|---|
| TRAIN | `[2014-01-01, validation_start)` | `validation_start` |
| VALIDATION | `[validation_start, test_start)` | `test_start` |
| TEST | `[test_start, sekarang]` | `data_end` (hari ini) |

Pada data saat eksperimen ini dibuat (`data_end` = 2026-08-03):
`validation_start` = 2025-01-01, `test_start` = 2026-01-01.

## Fitur (19, dari 21 fitur model classification)

Reuse penuh `feature_builder.build_features()` (fungsi asli, tidak diubah),
minus 2 kolom yang SELALU konstan pada `observation_on=installed_on`:
`log_days_since_installation` dan `installation_age_band` - umur pemasangan
di sini adalah **sumbu waktu model** (`duration_days`), bukan fitur input.

| Kelompok | Fitur |
|---|---|
| Kategorikal | `part_model_category`, `client_category` |
| Riwayat | `log_total_prior_events`, `log_prior_failure_count`, `has_prior_failure`, `log_prior_corrective_count`, `has_prior_corrective`, `log_days_since_last_corrective`, `log_prior_distinct_places` |
| Jendela waktu | `log_prior_corrective_30d`, `log_prior_failure_365d`, `log_prior_events_180d` |
| Lifecycle antar-siklus | `log_previous_cycle_lifetime_mean`, `has_previous_cycle` |
| Musiman (bulan instalasi) | `month_sin`, `month_cos` |
| Kondisi armada | `log_model_failures_90d`, `model_failure_rate_90d`, `log_model_fleet_size` |

Kategorikal di-one-hot-encode (RSF/CoxPH di `scikit-survival` tidak punya
native categorical handling seperti CatBoost); encoder di-fit HANYA di TRAIN
dan disimpan di `artifacts/encoder.joblib`.

## Model

- **Random Survival Forest** (`sksurv.ensemble.RandomSurvivalForest`) -
  model utama/`primary_model`.
- **Cox Proportional Hazards** (`sksurv.linear_model.CoxPHSurvivalAnalysis`,
  ridge kecil `alpha=0.1`) - baseline pembanding sederhana.

Keduanya dilatih dan dilaporkan berdampingan (pola yang sama seperti
perbandingan LogReg+RF pada model scrap di root), tanpa pencarian
hyperparameter besar-besaran - tujuannya membuktikan formulasi survival,
bukan memeras skor.

## Evaluasi (dua lapis, TIDAK dicampur)

**Lapis 1 - native survival** (dari `t=0=installed_on`, cara standar
survival dievaluasi): C-index (Harrell), Integrated Brier Score, Brier score
& time-dependent AUC pada horizon 30/60/90/120 hari, dihitung
`evaluate.py`. Horizon yang melebihi follow-up split dilaporkan sebagai
"tidak dapat dihitung", tidak dipaksakan.

**Lapis 2 - perbandingan adil dengan model classification production**
(supaya tidak membandingkan C-index vs ROC-AUC secara naif):

1. Pinjam baris TEST classification PERSIS sama lewat `train.build_dataset()`
   di root (read-only, tidak pernah dipakai fitting).
2. Untuk tiap baris (satu PART pada satu titik umur `A`): hitung
   `P(fail<=30 hari | selamat sampai umur A) = 1 - S(A+30)/S(A)` dari kurva
   survival model ini (fit di `t=0=installed_on`, fitur baseline instalasi -
   BUKAN fitur yang di-refresh ke tanggal snapshot).
3. Skor itu dievaluasi dengan `training_utils.full_metrics()`/
   `capacity_metrics()` di root (fungsi asli, kapasitas sama
   `config.FAILURE_CAPACITY_PER_MONTH`) - ROC-AUC, PR-AUC, Brier,
   Precision/Recall@kapasitas, persis definisi yang dipakai `train.py`
   classification.

Perbandingan ini adil dari sisi horizon/populasi/label/definisi metrik,
tapi TETAP memberi model classification keuntungan struktural (fiturnya
lebih segar - lihat "Keterbatasan" di atas). Disebutkan eksplisit, tidak
disamarkan.

## Arti C-index yang benar

> C-index adalah probabilitas bahwa model mengurutkan sepasang lifecycle
> yang bisa dibandingkan dalam urutan risiko/waktu failure yang BENAR.

C-index 0,80 **bukan** "80% akurat menebak tanggal kerusakan" - itu
interpretasi yang salah. Model survival tidak pernah memperkirakan tanggal
pasti; ia memberi kurva probabilitas bertahan (`S(t)`) dan risiko dalam
horizon (`risk_30d`, dst).

## Struktur

```
survival_model/
├── README.md
├── requirements.txt        # scikit-survival + numexpr (terpisah dari root)
├── build_dataset.py        # DB -> lifecycle eligible -> fitur -> split, + validasi data
├── train.py                 # latih RSF + Cox PH, simpan artifacts/
├── evaluate.py               # Lapis 1 + Lapis 2 evaluasi, tulis reports/
├── predict.py                 # CLI: python predict.py <item_id>
├── src/
│   ├── lifecycle_builder.py   # cohort filter + aturan censoring per-split
│   ├── features.py             # wrapper reuse feature_builder + one-hot encoder
│   └── utils.py                 # split bounds, evaluasi kurva S(t)
├── artifacts/                   # models.joblib, encoder.joblib, metadata.json
└── reports/                     # data_validation.md, evaluation_report.md
```

## Cara pakai

```bash
pip install --no-deps scikit-survival numexpr   # lihat requirements.txt untuk alasan --no-deps
python build_dataset.py    # cek dataset + validasi data
python train.py             # latih RSF + Cox PH
python evaluate.py           # Lapis 1 + Lapis 2, tulis reports/evaluation_report.md
python predict.py <item_id>   # prediksi satu PART aktif
```

Semua script membaca database yang SAMA (read-only, lewat `data_reader.py`
di root, tidak diubah) dan mengikuti pola import lintas-folder yang sudah
ada di `scripts/run_pipeline.py`.

## Catatan teknis: kenapa `duration_days` dibulatkan ke hari bulat

`RandomSurvivalForest` menyimpan satu titik kurva survival PER waktu unik DI
SETIAP leaf node di SETIAP pohon. `duration_days` yang dihitung langsung dari
selisih timestamp mentah (presisi jam/menit) membuat grid waktu unik meledak
sampai ribuan titik berbeda - percobaan pertama menghasilkan artifact model
**>4 GiB** dan proses prediksi yang hang. Presisi jam/menit itu sendiri tidak
berarti apa pun secara bisnis untuk "berapa lama PART bertahan", jadi
`duration_days` dibulatkan ke hari bulat (`src/lifecycle_builder.py`) -
mengecilkan grid waktu unik dari ~3.069 ke ~1.982 titik dan artifact model ke
~534 MB, **tanpa** menurunkan C-index (malah sedikit naik: 0,8065->0,8078 di
VALIDATION). `min_samples_leaf`/`min_samples_split` juga dinaikkan sedikit
(RSF_PARAMS di `train.py`) untuk mengecilkan lebih lanjut tanpa kehilangan
akurasi.

## Catatan teknis: prediksi dipaksa single-thread

`RandomSurvivalForest` yang di-unpickle di proses baru lalu diminta
`predict_survival_function()` dengan `n_jobs=-1` (default saat training)
terbukti membuat proses **hang tanpa error** saat `loky` mencoba membongkar
worker pool-nya - komputasinya sendiri selesai dalam hitungan milidetik,
hanya exit proses yang macet. `evaluate.py` dan `predict.py` memaksa
`model.n_jobs = 1` setelah `joblib.load()` untuk menghindari ini. Training
(`train.py`) tidak terpengaruh dan tetap memakai `n_jobs=-1`.

---

## Hasil

Dari training pada data s/d `data_end` = 2026-08-03 11:07:22.

### 1-2. Lifecycle & event/censored

23.927 lifecycle cohort (`is_initial_model_cohort`, durasi positif) ->
**20.116 lifecycle eligible** (84,1%) setelah aturan censoring per-split -
lihat `reports/data_validation.md` untuk rincian lengkap.

| Split | Baris | Event (failure) | Censored | Base rate |
|---|---|---|---|---|
| TRAIN | 14.980 | 3.041 | 11.939 | 20,3% |
| VALIDATION | 2.316 | 385 | 1.931 | 16,6% |
| TEST | 2.820 | 370 | 2.450 | 13,1% |

Base rate menurun dari TRAIN ke TEST - **bukan berarti fleet makin aman**,
tapi karena TEST berisi lifecycle yang baru dimulai (installed_on >=
2026-01-01) dengan follow-up jauh lebih pendek (maks 214 hari) dibanding
TRAIN (bisa sampai ribuan hari) - lebih sedikit waktu untuk failure
terjadi/tercatat. Perbandingan base rate antar split TIDAK apple-to-apple
karena alasan ini, dicatat sebagai keterbatasan.

### 3. Fitur

19 fitur (2 kategorikal one-hot + 14 numerik riwayat/lifecycle/musiman + 3
kondisi armada) - daftar lengkap di bagian "Fitur" di atas.

### 4. Model

Random Survival Forest (`n_estimators=100, min_samples_split=40,
min_samples_leaf=30`) sebagai model utama, Cox PH (`alpha=0.1`) sebagai
baseline pembanding.

### 5. C-index (Lapis 1, native survival, dari t=0=installed_on)

| Model | VALIDATION | TEST |
|---|---|---|
| Random Survival Forest | 0,8078 | 0,8051 |
| Cox PH | 0,7706 | 0,8149 |

Keduanya mengalahkan tebakan acak (0,5) dengan jelas - PROOF OF CONCEPT
bahwa formulasi survival BISA memisahkan lifecycle berumur pendek dari yang
panjang secara wajar.

### 6. Metrik survival lain

| Model | Split | Integrated Brier Score | Brier 30d | Brier 60d | Brier 90d | Brier 120d |
|---|---|---|---|---|---|---|
| RSF | VALIDATION | 0,0771 | 0,0652 | 0,0767 | 0,0806 | 0,0829 |
| RSF | TEST | 0,0801 | 0,0810 | 0,0818 | 0,0795 | 0,0771 |
| Cox PH | VALIDATION | 0,0921 | 0,0754 | 0,0927 | 0,0959 | 0,1003 |
| Cox PH | TEST | 0,0868 | 0,0843 | 0,0884 | 0,0867 | 0,0861 |

Time-dependent AUC (RSF, TEST): 30d=0,838, 60d=0,868, 90d=0,882, 120d=0,905 -
lengkap di `reports/evaluation_report.md`.

### 7. Risk performance 30/60/90/120 hari & 8. Perbandingan fair dengan existing (Lapis 2)

37.923 dari 38.451 baris TEST classification (211 hari window, kapasitas
200/bulan) cocok dengan lifecycle survival dan dievaluasi dengan
`P(fail<=30 hari | selamat sampai umur A) = 1-S(A+30)/S(A)`, dinilai dengan
`training_utils.full_metrics()` yang SAMA PERSIS dipakai `train.py`
classification:

| Model | PR-AUC | ROC-AUC | Recall@200/bln | Precision@200/bln | Brier |
|---|---|---|---|---|---|
| Random Survival Forest | 0,1558 | 0,7065 | 0,3142 | 0,1983 | 0,0214 |
| Cox PH | 0,1115 | 0,7063 | 0,2173 | 0,1372 | 0,0221 |
| **Classification (production, v2)** | **0,1607** | **0,8206** | **0,3359** | **0,2154** | 0,0215 |

Dibaca apa adanya: **classification model tetap lebih baik untuk tugas
operasional 30-hari** ini, di semua metrik ranking (ROC-AUC selisihnya
paling nyata: 0,71 vs 0,82) - sesuai prediksi di bagian "Keterbatasan"
(fitur classification di-refresh tiap 30 hari, fitur survival beku di
kondisi instalasi). Brier score nyaris identik (0,0214 vs 0,0215) - keduanya
sama-sama terkalibrasi cukup baik pada skala absolut, bedanya di
KEMAMPUAN MENGURUTKAN mana yang lebih berisiko. RSF tetap mengalahkan Cox
PH cukup jelas di sisi PR-AUC/Recall/Precision pada tugas ini.

### 9. Potensi leakage / isu data yang ditemukan

- **1.234 item (7,5%)** punya lifecycle di lebih dari satu split (cycle
  awal di TRAIN, cycle berikutnya di TEST/VALIDATION) - bukan leakage
  temporal, tapi potensi model "mengenali" identitas item lewat fitur
  `previous_cycle_lifetime_mean`. Tidak diperbaiki (lihat README bagian
  Leakage prevention).
- Base rate event menurun antar split (20,3% -> 16,6% -> 13,1%) - karena
  perbedaan panjang follow-up, bukan sinyal fleet membaik (lihat poin 1-2
  di atas).
- Model classification production (v2) dievaluasi ulang di sini via
  `train.build_dataset()` (bukan angka dari `metadata.json` production
  secara langsung untuk baris/populasi, tapi `promotion_comparison`
  candidate-nya) - konsisten dengan `evaluation_metrics.test` di
  `models/failure/CURRENT/metadata.json`.
- Riset lama (`db_om_preparation`) sudah pernah mencoba Cox PH/RSF/XGBoost
  AFT dan kalah dari model resmi (dicatat di README root) - eksperimen ini
  MEREPRODUKSI temuan itu dengan metodologi yang diperbaiki (censoring
  per-split, bukan snapshot classification dipaksa jadi survival), dan bisa
  menjelaskan SEBAGIAN alasannya secara eksplisit: keunggulan classification
  banyak berasal dari fitur yang lebih segar (di-refresh tiap 30 hari),
  bukan semata algoritma classification vs survival.

### 10. Rekomendasi akhir

**Survival model TIDAK direkomendasikan menggantikan model classification**
untuk tugas operasional "PART mana yang perlu diprioritaskan bulan ini" -
classification production tetap lebih baik pada SEMUA metrik ranking di
perbandingan adil Lapis 2 (poin 8).

**Survival model layak sebagai CHALLENGER/pelengkap**, bukan pengganti,
untuk pertanyaan yang classification TIDAK bisa jawab secara native:
"berapa lama PART ini diperkirakan bertahan" (median survival time, kurva
`S(t)` penuh) - berguna untuk perencanaan kapasitas/stok jangka menengah-
panjang, bukan prioritisasi harian/bulanan. C-index 0,80-0,81 pada tugas
native-nya (ranking seluruh lifetime) menunjukkan formulasinya SAH secara
metodologis, bukan sekadar re-hash percobaan lama yang gagal.

Kalau ke depannya ingin mendekati performa classification pada tugas
30-hari SAMBIL mempertahankan output survival (kurva S(t), median survival),
langkah lanjutan yang paling menjanjikan adalah landmarking/time-varying
covariates (fitur di-refresh berkala seperti classification, tapi target
tetap durasi-ke-event) - sengaja TIDAK dikerjakan di eksperimen ini karena
akan mengembalikan bentuk grid 30-harian yang ingin dihindari di awal, dan
di luar cakupan "jangan overengineering" untuk pembuktian formulasi awal ini.

---

Angka lengkap (semua split, semua metrik, evaluasi mentah): lihat
`artifacts/metadata.json` dan `reports/evaluation_report.md` (dihasilkan
`train.py`/`evaluate.py`, akan berubah kalau dijalankan ulang pada data
yang lebih baru).
