# Eksperimen: rekayasa ulang keseimbangan kelas (model kerusakan CatBoost)

Dijalankan 2026-08-22, atas permintaan eksplorasi ad-hoc - BUKAN bagian dari
Fase A-D restrukturisasi. Tidak menulis apa pun ke `models/failure/` -
seluruhnya jalur samping, tidak membuat versi baru maupun menyentuh `CURRENT`.

## Pertanyaan

Target 30-hari CatBoost sangat timpang (~1,5% positif di TRAIN, ~2,3% di
TEST). Ditangani sekarang lewat `auto_class_weights="Balanced"` (pembobotan
ulang loss function CatBoost, BUKAN mengubah datanya). Pertanyaannya: kalau
datanya sendiri direkayasa supaya tidak seimpang itu (undersample kelas
mayoritas, oversample kelas minoritas, atau kombinasi dengan class_weight),
apakah akurasi operasional naik atau turun?

## Metodologi

- `VALIDATION` dan `TEST` **tidak pernah** ikut direkayasa - selalu distribusi
  natural, supaya angka yang dilaporkan tetap mencerminkan populasi produksi
  sungguhan (bukan populasi buatan).
- Hanya `TRAIN` yang diubah: undersample = buang sebagian baris negatif acak;
  oversample = duplikasi baris positif (bootstrap, `replace=True`) sampai
  rasio target tercapai; negatif tetap utuh.
- Hyperparameter CatBoost SAMA PERSIS (`config.CATBOOST_PARAMS`) di semua
  varian kecuali `auto_class_weights` (dimatikan saat data sudah direkayasa,
  supaya tidak dobel-kompensasi - satu varian sengaja menguji dobel-kompensasi
  itu juga).
- Kalibrator isotonic dilatih ulang di `VALIDATION` natural untuk tiap varian
  (bukan cuma sekali), supaya kalibrasi selalu adil terhadap datanya sendiri.
- Metrik dihitung persis lewat `training.versioning.full_metrics()` - fungsi
  produksi yang sama dipakai `decide_promotion`, bukan rumus terpisah.
- Baseline di tabel = angka production yang sudah tercatat di
  `models/failure/v2/metadata.json` (direproduksi ulang di eksperimen ini
  untuk sanity-check - cocok persis sampai 4 desimal).

## Hasil (11 varian, diurutkan Recall@kapasitas menurun)

| Varian | Pos rate TRAIN | n TRAIN | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (`auto_class_weights`, data natural)** | 1,53% | 251.568 | 0,8211 | 0,1610 | 0,0215 | **0,3359** | **0,2154** |
| Tanpa `class_weight`, tanpa resample | 1,53% | 251.568 | 0,8175 | 0,1655 | 0,0213 | 0,3304 | 0,2118 |
| Undersample 25% (sendiri) | 25% | 15.408 | 0,8242 | 0,1497 | 0,0216 | 0,3315 | 0,2125 |
| Undersample 10% (sendiri) | 10% | 38.520 | 0,8268 | 0,1743 | 0,0212 | 0,3193 | 0,2047 |
| Oversample 10% | 10% | 275.240 | 0,8278 | 0,1559 | 0,0214 | 0,3115 | 0,1997 |
| Undersample 10% + `class_weight` | 10% | 38.520 | 0,8256 | 0,1612 | 0,0214 | 0,3071 | 0,1969 |
| Undersample 5% (sendiri) | 5% | 77.040 | 0,8287 | 0,1782 | 0,0212 | 0,3226 | 0,2068 |
| Undersample 50% / 1:1 (sendiri) | 50% | 7.704 | 0,8142 | 0,1492 | 0,0217 | 0,2871 | 0,1841 |
| Undersample 25% + `class_weight` | 25% | 15.408 | 0,8126 | 0,1331 | 0,0218 | 0,2849 | 0,1827 |
| Oversample 25% | 25% | 330.288 | 0,8122 | 0,1277 | 0,0217 | 0,2639 | 0,1692 |
| Oversample 50% / 1:1 | 50% | 495.432 | 0,8011 | 0,1000 | 0,0221 | 0,1973 | 0,1265 |

(Kolom Recall/Presisi@kapasitas dihitung pada kapasitas kerja tim
`config.FAILURE_CAPACITY_PER_MONTH=200`, diskalakan ke panjang window TEST -
metrik yang sama dipakai gerbang promosi produksi.)

## Temuan

1. **Tidak satu pun dari 11 varian mengalahkan baseline** di Recall@kapasitas
   maupun Presisi@kapasitas - dua metrik yang benar-benar menggerbang
   promosi (`decide_promotion`, TIDAK menggerbang di ROC-AUC). Yang paling
   dekat (undersample 25% sendirian) masih kalah tipis di keduanya.
2. **ROC-AUC/PR-AUC dan Recall/Presisi@kapasitas bergerak BERLAWANAN arah**
   pada undersample ringan (5-10%): ROC-AUC/PR-AUC naik jelas di atas
   baseline, tapi Recall/Presisi@kapasitas tetap turun. ROC-AUC/PR-AUC
   mengukur kualitas urutan di SELURUH rentang skor; Recall/Presisi@kapasitas
   cuma peduli ~200 baris teratas (~0,5% populasi) - baris negatif yang
   dibuang undersample rupanya termasuk yang membantu mempertajam urutan
   TEPAT di puncak, walau menyederhanakan pola di tengah distribusi. Ini
   bukti empiris langsung kenapa gerbang produksi sengaja tidak memakai
   ROC-AUC sendirian (lihat docstring `decide_promotion`).
3. **Kombinasi undersample + `class_weight` konsisten LEBIH BURUK** daripada
   masing-masing sendirian, di kedua rasio yang dicoba (10% dan 25%) -
   dobel-kompensasi (data sudah direkayasa DAN loss dibobot ulang) membuat
   model overcorrect ke kelas positif.
4. **Oversample (duplikasi) tidak menghindari masalah undersample, malah
   lebih buruk pada rasio yang sama** (mis. Recall@kapasitas oversample 10%
   = 0,3115 vs undersample 10% = 0,3193), dan memburuk drastis pada rasio
   tinggi - oversample 50% adalah **varian TERBURUK dari seluruh 11**
   (Recall@kapasitas 0,1973, hampir separuh baseline). Duplikasi baris
   membuat CatBoost menghafal baris yang sama berulang (overfit ke
   duplikat), bukan belajar pola baru dari informasi baru.

## Kesimpulan

`auto_class_weights="Balanced"` pada data natural (pendekatan production
sekarang) **tetap yang terbaik** untuk metrik yang benar-benar dipakai
mengevaluasi model ini. **Tidak direkomendasikan mengubah pendekatan
penanganan imbalance yang sekarang dipakai.**

## Yang TIDAK dicoba (bukan diabaikan, di luar cakupan eksperimen ini)

- SMOTE / oversample sintetis (bukan duplikasi bootstrap polos) - lebih rumit
  diterapkan karena fitur kategorikal (CatBoost butuh kategori asli, bukan
  interpolasi numerik SMOTE standar).
- Resample pada `class_weights` manual (bobot custom selain `"Balanced"`)
  atau `scale_pos_weight` custom.
- Rasio undersample/oversample di antara titik yang diuji (mis. 15%, 20%).

Kalau pertanyaan ini muncul lagi di masa depan: baca laporan ini dulu sebelum
mengulang - kesimpulan intinya (rekayasa data tidak menang di metrik gerbang)
kemungkinan besar masih berlaku kecuali targetnya (kapasitas kerja, definisi
horizon) ikut berubah.
