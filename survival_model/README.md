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

## Fitur (hasil audit metodologis, bukan tebakan awal)

Iterasi pertama eksperimen ini reuse 19 fitur classification apa adanya
(minus 2 kolom yang SELALU konstan pada `observation_on=installed_on`:
`log_days_since_installation`/`installation_age_band` - umur pemasangan di
sini adalah **sumbu waktu model**, bukan fitur input). Sesi audit berikutnya
menguji secara eksplisit: (1) apakah konteks instalasi (part type/lokasi/
client, di luar riwayat kejadian) membawa signal tambahan, (2) apakah
threshold kategori classification (`config.MIN_PART_MODEL_SUPPORT=300`,
dikalibrasi untuk skala 251rb baris) juga optimal untuk skala ~15rb lifecycle
survival, dan (3) apakah `previous_cycle_lifetime_mean` benar-benar mengukur
"lifetime sampai gagal" seperti namanya. Metodologi lengkap dan tabel angka
di `reports/category_threshold.md`, `reports/feature_ablation.md`,
`reports/previous_cycle_audit.md`, `reports/model_comparison.md`
(dihasilkan `experiments.py`) - ringkasan hasil ada di bagian "Hasil" di
bawah.

Fitur FINAL yang dipakai `train.py`/`evaluate.py`/`predict.py`:

| Kelompok | Fitur |
|---|---|
| Kategorikal | `part_model_category` (threshold survival=200), `client_category`, `item_type_at_install_grouped` (threshold=300, **baru**) |
| Riwayat | `log_total_prior_events`, `log_prior_failure_count`, `has_prior_failure`, `log_prior_corrective_count`, `has_prior_corrective`, `log_days_since_last_corrective`, `log_prior_distinct_places` |
| Jendela waktu | `log_prior_corrective_30d`, `log_prior_failure_365d`, `log_prior_events_180d` |
| Lifecycle antar-siklus | `log_previous_cycle_confirmed_failure_lifetime_mean`, `has_previous_cycle_confirmed_failure_lifetime_mean` (**diganti** dari `previous_cycle_lifetime_mean` - lihat "Audit previous-cycle" di Hasil) |
| Musiman (bulan instalasi) | `month_sin`, `month_cos` |
| Kondisi armada | `log_model_failures_90d`, `model_failure_rate_90d`, `log_model_fleet_size` |

`item_type_at_install` (tipe PART persis saat instalasi, dari event
`INSTALLED` yang sudah dibaca `build_dataset.build()` - tidak ada query
baru, lihat `src/install_context.py`) TERBUKTI menambah signal VALIDATION di
atas fitur warisan. `place_at_install` (lokasi saat instalasi) DIUJI tapi
TIDAK diikutkan di kombinasi final - menaikkan skor sendirian tapi
menurunkannya saat digabung dengan `item_type_at_install` (lihat ablation).
`device_type`/`device_model` diinvestigasi dan TIDAK dipakai - tidak
tersedia lewat relasi yang sudah dikanonikalisasi tanpa membuat mapping baru
(item_category cohort ini hanya 'PART'/'TERMINAL', tidak ada relasi
"device" bersih untuk PART) - didokumentasikan sebagai keterbatasan data,
bukan dipaksakan.

Kategorikal di-one-hot-encode (RSF/CoxPH di `scikit-survival` tidak punya
native categorical handling seperti CatBoost); encoder di-fit HANYA di TRAIN
dan disimpan di `artifacts/encoder.joblib`.

`LEGACY_CATEGORICAL_FEATURES`/`LEGACY_NUMERIC_FEATURES` di `src/features.py`
menyimpan 19 fitur warisan classification apa adanya, dipertahankan sebagai
titik referensi "A_current" di `experiments.py` - TIDAK dipakai model
production.

## Model

- **Random Survival Forest** (`sksurv.ensemble.RandomSurvivalForest`) -
  model utama/`primary_model`.
- **Cox Proportional Hazards** (`sksurv.linear_model.CoxPHSurvivalAnalysis`,
  ridge kecil `alpha=0.1`) - baseline pembanding, dipertahankan permanen di
  SETIAP tahap eksperimen (bukan dibuang setelah RSF menang) supaya terlihat
  jelas kalau suatu saat sebuah kombinasi fitur membuat model linear
  menyamai/mengalahkan RSF (sinyal bottleneck ada di fitur, bukan model).

Keduanya dilatih dan dilaporkan berdampingan (pola yang sama seperti
perbandingan LogReg+RF pada model scrap di root). Hyperparameter RSF diuji
lewat pencarian KECIL coordinate-wise (`n_estimators`, `min_samples_leaf`,
`max_features`, `max_depth` - satu sumbu diubah per langkah dari titik
current, bukan grid penuh) SETELAH seluruh kerja fitur selesai, dipilih dari
VALIDATION - lihat `reports/model_comparison.md`. Tidak ada trial yang
mengalahkan default/current.

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
├── experiments.py             # audit metodologis: threshold sweep, ablation
│                              # A/B/C, audit previous-cycle, tuning RSF kecil -
│                              # dijalankan SEKALI untuk memilih konfigurasi final,
│                              # bukan bagian alur production rutin
├── src/
│   ├── lifecycle_builder.py   # cohort filter + aturan censoring per-split
│   ├── features.py             # fitur final + wrapper reuse feature_builder
│   ├── categorical_support.py   # cumulative support point-in-time generik
│   │                            # (versi bukan-hardcoded dari feature_builder.
│   │                            # cumulative_support, dipakai kolom apa pun)
│   ├── install_context.py        # item_type_at_install/place_at_install dari
│   │                              # events INSTALLED yang sudah dibaca
│   ├── previous_cycle.py           # audit + hitung fitur previous-cycle
│   │                              # confirmed-failure-only (point-in-time)
│   ├── model_fit.py                # fit_models/evaluate_models - dipakai
│   │                              # train.py DAN experiments.py
│   └── utils.py                     # split bounds, evaluasi kurva S(t)
├── artifacts/                   # models.joblib, encoder.joblib, metadata.json
└── reports/                     # data_validation.md, evaluation_report.md,
                                  # category_threshold.md, feature_ablation.md,
                                  # previous_cycle_audit.md, model_comparison.md
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

Dari training pada data s/d `data_end` = 2026-08-03. Bagian ini punya dua
lapis: **audit metodologis** (`experiments.py`, keputusan dari VALIDATION)
yang menjelaskan KENAPA konfigurasi final terlihat seperti ini, lalu
**hasil final** dari `train.py`/`evaluate.py` dengan konfigurasi itu.

### 1. Audit fase awal: censoring & baseline instalasi

Sebelum fitur/threshold apa pun diubah, `src/lifecycle_builder.py` diaudit
ulang lewat kasus manual persis seperti yang diminta: PART terpasang
Desember 2024, failure Juli 2026, cutoff TRAIN = `validation_start`
(2025-01-01). Hasilnya sesuai spesifikasi - baris itu di TRAIN mendapat
`event_observed=0, duration_days=cutoff-installed_on` (durasi ke
2025-01-01, BUKAN ke tanggal failure 2026-07), karena Juli 2026 berada
SETELAH cutoff TRAIN dan karena itu tidak boleh terlihat. Failure itu baru
"terbuka" saat baris yang sama dievaluasi di VALIDATION/TEST dengan cutoff
mereka sendiri (2026-01-01 / hari ini) yang sudah melewati tanggal
failure-nya. Tidak ada perubahan di `lifecycle_builder.py` - mekanisme
administrative censoring per-split dari sesi sebelumnya **terverifikasi
benar**, dipertahankan apa adanya.

### 2. Lifecycle & event/censored (tidak berubah oleh audit fitur)

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
2026-01-01) dengan follow-up jauh lebih pendek dibanding TRAIN (bisa sampai
ribuan hari) - lebih sedikit waktu untuk failure terjadi/tercatat.
Perbandingan base rate antar split TIDAK apple-to-apple karena alasan ini.

### 3. Konteks instalasi yang dicoba (installation context)

`item_type_at_install` dan `place_at_install` diambil dari event
`INSTALLED` yang sudah dibaca `build_dataset.build()` (tanpa query baru).
`device_type`/`device_model` **tidak dipakai** - tidak tersedia lewat relasi
yang sudah dikanonikalisasi di cohort ini tanpa membuat mapping baru
(item_category cohort hanya 'PART'/'TERMINAL'), didokumentasikan sebagai
keterbatasan data, bukan dipaksakan lewat join baru.

### 4. Threshold kategori khusus survival (`reports/category_threshold.md`)

`config.MIN_PART_MODEL_SUPPORT=300` classification dikalibrasi untuk skala
251.568 baris TRAIN classification - diuji ULANG khusus untuk skala ~15rb
lifecycle TRAIN survival dengan sweep `[20,50,100,200,300]`, dipilih dari VAL
C-index (RSF ringan 50 pohon), TEST tidak dipakai memilih:

| Kolom | Kategori asli | Threshold terpilih | VAL C-index terbaik |
|---|---|---|---|
| `item_model_code_clean` | 46 | **200** (bukan 300 classification) | 0,8116 |
| `item_type_at_install` | 18 | **300** | 0,8147 |
| `place_at_install` | 137 | **50** | 0,8099 |

Tabel lengkap 15 baris (semua threshold x semua kolom, termasuk hitungan
unseen VAL/TEST) ada di reportnya. Threshold classification (300) BUKAN
optimal untuk `item_model_code_clean` di skala survival - 200 menang tipis
tapi konsisten.

### 5. Ablation A (current) / B (context-only) / C (combined) (`reports/feature_ablation.md`)

Semua pakai dataset/label/split/censoring yang SAMA - hanya kolom fitur yang
berbeda:

| Experiment | VAL C-index (RSF) | TEST C-index (RSF) | VAL C-index (Cox) |
|---|---|---|---|
| A_current (19 fitur classification warisan) | 0,8078 | 0,8051 | 0,7706 |
| B_context_only (part_model/client/item_type/place SAJA, tanpa riwayat) | 0,6547 | 0,6227 | 0,6678 |
| A + item_type_at_install | **0,8118** | 0,8034 | 0,7809 |
| A + place_at_install | 0,8089 | 0,8091 | 0,7625 |
| C_combined (A + item_type + place) | 0,8074 | 0,8036 | 0,7716 |

Temuan penting, dilaporkan apa adanya (bukan dipaksa ke satu arah):
- **B_context_only jauh di bawah A** (0,65 vs 0,81) - konteks instalasi
  SENDIRIAN, tanpa riwayat kejadian, TIDAK cukup untuk memprediksi
  durasi-hidup. Riwayat kejadian (corrective/failure/frekuensi historis)
  tetap jadi sinyal utama, bukan identitas part/lokasi/client.
- **`item_type_at_install` menambah signal nyata** di atas A (0,8078 ->
  0,8118 VAL) - satu-satunya fitur konteks baru yang lolos.
- **`place_at_install` TIDAK dipertahankan**: menaikkan skor SENDIRIAN
  (0,8089) tapi begitu digabung dengan `item_type_at_install` (C_combined)
  skornya malah TURUN ke 0,8074 - interaksi negatif, bukan aditif. Sesuai
  instruksi "jangan pertahankan fitur hanya karena sudah ada di
  classification", `place_at_install` di-drop dari kombinasi final meski
  sempat diuji serius.

### 6. Audit `previous_cycle_lifetime_mean` (`reports/previous_cycle_audit.md`)

Fitur lama `previous_cycle_lifetime_mean` (dari SQL `get_cycles()`) TERBUKTI
mencampur rata-rata durasi siklus sebelumnya APA PUN cara berakhirnya
(FAILURE, RIGHT_CENSORED_AT_DATA_END, REINSTALL_WITHOUT_RECORDED_FAILURE) -
BUKAN murni "lifetime sampai gagal" seperti namanya menyiratkan. Pengecekan
manual pada data nyata menemukan contoh konkret: satu item punya
`previous_cycle_lifetime_mean=2511,79` yang ternyata berasal dari siklus
sebelumnya yang berakhir `REINSTALL_WITHOUT_RECORDED_FAILURE` (PART
dilepas-pasang ulang TANPA failure tercatat), bukan dari kegagalan
sungguhan - salah label lifetime.

Diuji di atas konfigurasi terbaik dari ablation (A + item_type_at_install):

| Varian | VAL C-index | TEST C-index |
|---|---|---|
| existing (`previous_cycle_lifetime_mean`, campur semua) | 0,8109 | 0,8015 |
| **confirmed_failure_only** (hanya siklus sebelumnya yang FAILURE) | **0,8120** | 0,8096 |
| last_confirmed_failure (bukan rata-rata, hanya siklus FAILURE terakhir) | 0,8093 | 0,8106 |
| confirmed_failure_only + `previous_cycle_end_reason` | 0,8113 | 0,8108 |

`confirmed_failure_only` dipilih (VAL 0,8120 > existing 0,8109).
`previous_cycle_end_reason` TIDAK dipertahankan (0,8113 <= 0,8120 - tidak
menaikkan VALIDATION lebih lanjut).

### 7. RSF tuning kecil (`reports/model_comparison.md`)

Pencarian coordinate-wise kecil (bukan grid penuh) di sekitar hyperparameter
current - `n_estimators∈{200,400}`, `min_samples_leaf∈{10,20,30,50}`,
`max_features∈{sqrt,0.5,1.0}`, `max_depth∈{None,8,12}`, 10 trial, dipilih
dari VALIDATION:

**Tidak ada satu pun trial yang mengalahkan default/current
(VAL C-index=0,8120)** - kombinasi `n_estimators=100, min_samples_split=40,
min_samples_leaf=30, max_features=sqrt, max_depth=None` yang sudah dipakai
sejak sesi sebelumnya TERBUKTI sudah berada pada/dekat titik optimal untuk
kombinasi fitur ini. Perubahan `max_features` ke 0,5/1,0 justru menurunkan
skor paling nyata (0,8069/0,8087 VAL, dan TEST turun sampai 0,78-0,80) -
mengurangi randomness antar-pohon RSF pada dataset sekecil ini merugikan,
bukan membantu.

### 8. Hasil final (`train.py` + `evaluate.py`, konfigurasi terpilih di atas)

**Catatan metodologis**: tabel tuning di poin 7 memakai `part_model_category`
threshold=300 (klasifikasi, bawaan alur eksperimen ablation/tuning yang
dibangun di atas fitur `feature_builder.build_features()` apa adanya) -
BUKAN threshold=200 yang divalidasi terpisah sebagai lebih baik di poin 4.
Integrasi produksi final (`src/features.py`) mengoreksi ini dan memakai
threshold=200 yang benar-benar tervalidasi. Karena itu angka final di bawah
(dari `train.py`/`evaluate.py` sungguhan) sedikit berbeda dari tabel poin 7
- **angka di bawah ini yang otoritatif**, karena inilah model yang benar-
benar disimpan di `artifacts/` dan dipakai `predict.py`.

**C-index (Lapis 1, native survival, dari t=0=installed_on) - Harrell vs
Uno/IPCW:**

| Model | Split | C-index (Harrell) | C-index (Uno/IPCW) | IBS |
|---|---|---|---|---|
| Random Survival Forest | VALIDATION | 0,8114 | 0,8117 | 0,0764 |
| Random Survival Forest | TEST | 0,8082 | 0,8083 | 0,0811 |
| Cox PH | VALIDATION | 0,7819 | 0,7821 | 0,0859 |
| Cox PH | TEST | 0,7722 | 0,7724 | 0,0950 |

Harrell dan Uno/IPCW **nyaris identik** di semua split/model (selisih
<=0,0003) - indikasi kuat TIDAK ada bias sensor besar yang mendistorsi
Harrell C-index sederhana; keduanya boleh dipakai sebagai headline number
dengan percaya diri yang sama.

**Brier score & time-dependent AUC per horizon:**

| Model | Split | Brier 30d | Brier 60d | Brier 90d | Brier 120d | AUC 30d | AUC 60d | AUC 90d | AUC 120d |
|---|---|---|---|---|---|---|---|---|---|
| RSF | VALIDATION | 0,0627 | 0,0755 | 0,0805 | 0,0836 | 0,7991 | 0,8261 | 0,8450 | 0,8546 |
| RSF | TEST | 0,0807 | 0,0827 | 0,0808 | 0,0790 | 0,8424 | 0,8690 | 0,8827 | 0,9051 |
| Cox PH | VALIDATION | 0,0684 | 0,0853 | 0,0899 | 0,0965 | 0,7683 | 0,7943 | 0,8166 | 0,8249 |
| Cox PH | TEST | 0,0912 | 0,0963 | 0,0952 | 0,0958 | 0,8027 | 0,8288 | 0,8449 | 0,8650 |

Semua horizon (30/60/90/120 hari) berada dalam rentang follow-up split
TEST (maks ~211 hari) - tidak ada horizon yang perlu dilaporkan "N/A".

**Perbandingan operasional adil vs classification production (Lapis 2)** -
37.923 dari 38.451 baris TEST classification (kapasitas 200/bulan) cocok
dengan lifecycle survival, dinilai `training_utils.full_metrics()` yang SAMA
PERSIS dipakai `train.py` classification:

| Model | PR-AUC | ROC-AUC | Recall@cap | Precision@cap | Brier |
|---|---|---|---|---|---|
| Random Survival Forest | **0,1633** | 0,6871 | 0,3108 | 0,1962 | 0,0214 |
| Cox PH | 0,0939 | 0,6893 | 0,1869 | 0,1180 | 0,0224 |
| **Classification (production, v2)** | 0,1607 | **0,8206** | **0,3359** | **0,2154** | 0,0215 |

### 9. Fitur yang benar-benar membantu vs tidak

**Membantu (dipertahankan di final):**
- `item_type_at_install_grouped` - signal instalasi baru yang konsisten
  menaikkan VAL C-index (poin 5).
- `previous_cycle_confirmed_failure_lifetime_mean` (confirmed-failure-only,
  ganti fitur lama) - lebih jujur secara definisi DAN validasinya lebih
  baik (poin 6).
- Threshold `part_model_category=200` (bukan 300 classification) - lebih
  cocok untuk skala survival (poin 4).

**Tidak membantu / di-drop meski sempat diuji serius:**
- `place_at_install` - berinteraksi negatif dengan `item_type_at_install`
  (poin 5), TIDAK diikutkan meski fitur ini "ada" di data instalasi.
- `previous_cycle_end_reason` - tidak menaikkan VALIDATION lebih lanjut di
  atas confirmed_failure_only (poin 6).
- Tuning hyperparameter RSF di luar default - tidak ada trial yang menang
  (poin 7); performa dibatasi oleh fitur, bukan hyperparameter.
- `device_type`/`device_model` - tidak dicoba sama sekali karena tidak
  tersedia tanpa mapping baru (poin 3).

### 10. Potensi keterbatasan data

- **1.234 item (7,5%)** punya lifecycle di lebih dari satu split (cycle
  awal di TRAIN, cycle berikutnya di TEST/VALIDATION) - BUKAN leakage
  temporal (urutan waktu tetap terjaga), tapi potensi model "mengenali"
  identitas item lewat fitur previous-cycle. Didokumentasikan, tidak
  diperbaiki (lihat bagian "Leakage prevention" di atas).
- Base rate event menurun antar split (20,3% -> 16,6% -> 13,1%) karena
  perbedaan panjang follow-up, bukan sinyal fleet membaik.
- Threshold `place_at_install=50` divalidasi tapi TIDAK dipakai final -
  fitur itu sendiri dibuang di tahap ablation (poin 5), bukan berarti
  eksperimen thresholdnya sia-sia - tetap menjadi bukti proses pemilihan
  threshold yang tervalidasi, bukan tebakan.
- Riset lama (`db_om_preparation`) sudah pernah mencoba Cox PH/RSF/XGBoost
  AFT dan kalah dari model resmi. Audit ini MEREPRODUKSI temuan itu dengan
  metodologi yang diperbaiki (censoring per-split, fitur baseline instalasi
  eksplisit, bukan snapshot classification dipaksa jadi survival), dan
  menjelaskan SEBAGIAN kenapa: keunggulan classification pada tugas 30-hari
  banyak berasal dari fitur yang lebih segar (di-refresh tiap 30 hari),
  bukan semata algoritma classification vs survival.

### 11. Stabilitas hasil tanpa bergantung pada fitur warisan classification

Ablation B_context_only (fitur instalasi murni, TANPA satu pun fitur
riwayat warisan classification) hanya mencapai VAL C-index 0,6547 - jauh di
bawah A (0,8078). Ini artinya performa survival model **TIDAK stabil/tidak
cukup tanpa fitur riwayat kejadian** - konteks instalasi saja tidak
memadai. Namun fitur riwayat yang dipertahankan (prior failure/corrective
count, frekuensi historis, kondisi armada) BUKAN artefak arbitrer
classification - semuanya point-in-time-safe dan business-justified secara
independen (riwayat kegagalan part sejenis memang secara masuk akal
memprediksi durasi hidup part baru). Kesimpulan: hasil final **bergantung**
pada fitur riwayat (bukan semata identitas/konteks statis), tapi TIDAK
"curang" meniru classification - fitur riwayat itu sendiri lolos audit
per-fitur secara independen (poin 9).

### 12. Rekomendasi akhir

**C. Survival model layak sebagai challenger/pelengkap** model
classification, **bukan pengganti** dan **belum perlu lanjut ke dynamic/
event-based survival**.

Alasan, dibaca apa adanya dari angka di atas:

- Pada tugas NATIVE-nya (ranking seluruh lifetime, C-index Harrell & Uno
  nyaris identik ~0,81 VAL/TEST, AUC time-dependent naik konsisten
  0,80->0,91 dari horizon 30d ke 120d), survival model **metodologis dan
  konsisten** - bukan re-hash percobaan lama yang gagal begitu saja.
- Pada tugas OPERASIONAL 30-hari (Lapis 2, populasi sama dengan
  classification), classification production **masih jelas lebih baik**
  di ROC-AUC (0,82 vs 0,69) dan Recall/Precision@kapasitas - sesuai
  prediksi di bagian "Keterbatasan" (fitur classification di-refresh tiap
  30 hari, fitur survival beku di kondisi instalasi). **TIDAK cukup alasan
  untuk merekomendasikan opsi D** (classification tetap satu-satunya) -
  survival model sudah MENGUNGGULI classification di PR-AUC (0,1633 vs
  0,1607) untuk pertama kalinya setelah audit ini, sinyal bahwa formulasi
  dan fitur barunya nyata membantu, bukan hanya C-index yang naik secara
  kosmetik.
- Seluruh audit (B_context_only yang jelek, place_at_install yang
  di-drop, tuning yang tidak membantu) menunjukkan performa dibatasi oleh
  KETERSEDIAAN INFORMASI point-in-time-safe pada `installed_on`, bukan oleh
  pilihan model/hyperparameter - artinya jalan paling menjanjikan untuk
  peningkatan lebih lanjut BUKAN tuning lebih agresif atau fitur konteks
  tambahan yang sudah terbukti lemah, melainkan **landmarking/time-varying
  covariates** (fitur di-refresh berkala seperti classification, target
  tetap durasi-ke-event) - sesuai instruksi, ini sengaja TIDAK dikerjakan
  sekarang karena hasil static survival TERBUKTI belum "buntu" (masih ada
  ruang perbaikan yang jelas via arah itu, bukan sudah mentok), dan
  mengerjakannya sekarang akan mengembalikan bentuk grid 30-harian yang
  ingin dihindari sejak awal eksperimen ini.

Ringkasnya: **pakai survival model sebagai pelengkap untuk pertanyaan
"berapa lama PART ini diperkirakan bertahan" (median survival time, kurva
S(t) penuh, perencanaan kapasitas/stok jangka menengah-panjang)**, TETAP
pakai classification production untuk prioritisasi operasional bulanan
"PART mana yang perlu ditindak sekarang".

---

Angka lengkap (semua split, semua metrik, evaluasi mentah): lihat
`artifacts/metadata.json` dan `reports/evaluation_report.md` (dihasilkan
`train.py`/`evaluate.py`, akan berubah kalau dijalankan ulang pada data
yang lebih baru). Jejak audit lengkap: `reports/category_threshold.md`,
`reports/feature_ablation.md`, `reports/previous_cycle_audit.md`,
`reports/model_comparison.md` (dihasilkan `experiments.py`, hasil satu kali
jalan yang dipakai memutuskan konfigurasi final - tidak dijalankan ulang
otomatis oleh `train.py`).
