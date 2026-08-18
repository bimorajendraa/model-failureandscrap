# production_ml

Versi minimal dan siap pakai dari pipeline prediksi PART: **training,
retraining, dan prediction**. Tidak ada notebook, EDA, profiling, visualisasi,
ablation study, atau eksperimen di sini - semua itu tetap di
`db_om_preparation/` sebagai referensi research.

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
├── models/
│   ├── failure/        # model KERUSAKAN
│   │   ├── CURRENT     # versi yang dipakai production
│   │   └── v1/         # model.cbm, calibrator.joblib, metadata.json
│   └── scrap/          # model SCRAP
│       ├── CURRENT
│       └── v1/         # model.joblib, metadata.json
├── requirements.txt
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

Dibandingkan terhadap base rate validasi (frekuensi kerusakan historis
sungguhan), bukan ambang karangan:

Satu angka yang mengatur semuanya, di `config.py`:

```python
FAILURE_CAPACITY_PER_MONTH = 200   # berapa PART/bulan yang sanggup diprioritaskan
```

Seluruh PART aktif diurutkan menurut risiko, lalu sebanyak kapasitas itulah
yang masuk `HIGH`. Karena tiap PART dinilai ulang tiap 30 hari, jumlah PART di
daftar `HIGH` pada satu saat sama dengan beban kerja per bulan. Hasilnya pada
16.877 PART aktif: **200 HIGH, 200 MEDIUM, 16.477 LOW** - tepat sesuai
kapasitas.

Pilihan lain, diukur di data uji 2026:

| Kapasitas/bln | Presisi | Tertangkap | Berapa kali lebih tepat |
|---|---|---|---|
| 50 | 29,4% | 145 dari 902 | 12,5x |
| 100 | 20,3% | 267 dari 902 | 8,6x |
| **200** (dipakai) | **16,6%** | **329 dari 902** | **7,1x** |
| 400 | 7,4% | 496 dari 902 | 3,2x |
| 800 | 7,4% | 633 dari 902 | 3,2x |

200/bulan dipilih karena **setara dengan aturan lama yang sudah tervalidasi di
research** (>=3x base rate validasi: presisi 16,6%, recall 36,6%). Jadi
perilakunya tidak berubah - yang berubah cara menyetelnya, dari kelipatan
statistik menjadi angka kapasitas yang bisa dibicarakan dengan tim operasional.

### Kenapa batas kelompok memakai skor mentah

Batas `HIGH`/`MEDIUM` dibandingkan terhadap **skor mentah model**, bukan
probabilitas terkalibrasi yang ditampilkan ke pengguna. Alasannya teknis:
kalibrator menghasilkan dataran, sehingga 16.877 PART hanya menempati sekitar
30 nilai probabilitas berbeda - jumlah PART yang tertandai melompat dari 97
langsung ke 303, tidak ada nilai di antaranya. Skor mentah punya ribuan nilai
berbeda dengan **urutan yang sama persis**, jadi batas bisa ditaruh tepat
sesuai kapasitas. Probabilitas terkalibrasi tetap yang dilaporkan, karena itu
angka yang bermakna untuk dibaca.

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

Tidak ada angka yang dikarang. Semuanya dari hasil research yang sudah diuji:

| Konstanta | Nilai | Asal |
|---|---|---|
| Hyperparameter CatBoost | depth 4, lr 0,03, l2 10 | pencarian hyperparameter, notebook 05 |
| Jumlah iterasi | 200 (tetap, bukan early stopping) | early stopping berbasis AUC bisa berhenti sangat prematur pada validasi yang positifnya sedikit |
| Horizon target | 30 hari | model resmi research |
| Ambang dukungan tipe PART | 300 observasi | rare-category ablation |
| Batas kelompok umur | 91/181/366/731/1461 hari | definisi fitur SQL research |
| Ambang risiko | 3x dan 1x base rate validasi | diuji pada data uji 2026 |
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
