# Langkah A — Baseline error sisa umur & kalibrasi kurva (RSF, SEBELUM perbaikan)

Rencana pengguna secara eksplisit membingkai ulang tujuan RSF: bukan ranking
(PR-AUC/Recall@kapasitas), tapi **seberapa dekat median/kurva S(t) dengan
kejadian nyata**. Laporan ini adalah Langkah A dari rencana itu - baseline
WAJIB sebelum eksperimen apa pun, supaya perubahan berikutnya bisa diukur
sebagai perbaikan sungguhan, bukan cuma terlihat rapi.

## Metodologi

- Populasi: TEST landmark (holdout, 4.890 baris, 412 event_observed=True),
  DIPECAH dua: (a) seluruh TEST, dan (b) subset `landmark_source=="ANCHOR"`
  (2.041 baris) - lebih mirip populasi SERVING (PART diskor "sekarang" terlepas
  ada/tidaknya event operasional baru-baru ini, sama seperti
  `feature_builder.current_observations()` yang dipakai `score_batch()`/`predict()`;
  landmark INSTALL selalu age=0, tidak representatif untuk "kondisi PART aktif
  hari ini").
- MAE/bias median: HANYA baris `event_observed=True` DAN median terisi - durasi
  aktual cuma pasti diketahui untuk baris ini. Bias SIGNED (bukan absolute) -
  `+` berarti model TERLALU OPTIMIS (median lebih panjang dari kenyataan).
- Kalibrasi kurva MENTAH: mean S(d) prediksi vs proporsi empiris "masih hidup di
  hari d" pada grid 30/60/90/120/180/365 hari - definisi "masih hidup" cuma
  pasti diketahui untuk baris `duration>=d` (survive/censored setelah d) atau
  `event & duration<d` (gagal sebelum d); censored SEBELUM d dibuang (tidak
  diketahui).

## Hasil

### Median: bias OPTIMIS masif dan sistematis

| Subset | n event usable | % median null | MAE median | Bias median (signed) | % over-predict |
|---|---:|---:|---:|---:|---:|
| Seluruh TEST | 257 | 79.3% | 751.9 hari | **+751.9 hari** | **99.6%** |
| ANCHOR saja (mirip serving) | 18 | 87.8% | 1303.8 hari | **+1303.8 hari** | **100.0%** |

Hampir SELALU (99.6-100%) model memprediksi median jauh LEBIH PANJANG dari
durasi sebenarnya - bukan noise dua arah, tapi bias satu arah yang konsisten.
Satu-satunya item_type dengan cukup event untuk dilaporkan (MODULE READER,
n=10, subset ANCHOR): bias +1.773,7 hari, MAE 1.794 hari - arah dan besaran
sama dengan agregat, memperkuat bahwa ini bukan campuran beberapa kelompok yang
saling menutupi.

**Catatan kehati-hatian**: n usable di subset ANCHOR sangat kecil (18 baris) -
angka pastinya (1.303,8 hari) tidak presisi, tapi ARAH (selalu over-predict)
konsisten dengan subset TEST penuh (n=257) yang jauh lebih besar, jadi temuan
arahnya kuat walau besaran di ANCHOR harus dibaca sebagai indikasi, bukan angka
pasti.

### Kalibrasi kurva mentah: S(t) turun TERLALU LAMBAT dibanding kenyataan

| Horizon | mean S(d) prediksi (mentah) | proporsi empiris masih hidup | gap |
|---|---:|---:|---:|
| 30d | 0.9560 | 0.9281 | 0.028 |
| 60d | 0.9504 | 0.9104 | 0.040 |
| 90d | 0.9224 | 0.8457 | 0.077 |
| 120d | 0.9084 | 0.8140 | 0.094 |
| 180d | 0.7496 | 0.2862 | 0.464 (lihat catatan) |
| 365d | 0.6386 | 0.0000 | (lihat catatan) |

Gap TUMBUH monoton dari 30d ke 120d (0,028 -> 0,094) - **model konsisten
memprediksi survival lebih tinggi dari kenyataan, dan gap-nya makin besar
seiring horizon makin jauh**. Ini secara langsung MENJELASKAN bias median di
atas: kalau S(t) turun terlalu lambat, waktu yang dibutuhkan untuk mencapai
S=0,5 (median) jadi jauh lebih lama dari yang seharusnya.

**Catatan kehati-hatian penting untuk 180d/365d**: follow-up maksimum TEST
cuma 214 hari (`metadata.json`'s `max_followup_days`) - di atas itu, populasi
"usable" (baris yang bisa dipastikan hidup/mati di hari itu) SECARA STRUKTURAL
condong ke baris yang gagal AWAL (baris yang bertahan lama belum tentu sudah
"terkonfirmasi hidup" di 180/365 hari saat data_end dipotong). Proporsi 0,0 di
365d BUKAN bukti "semua PART pasti mati di 365 hari" - itu artefak dari tidak
ada satu pun baris TEST yang punya follow-up cukup panjang untuk membuktikan
survival di titik itu. Angka 180d/365d dilaporkan sebagai indikasi arah (masih
konsisten dengan tren 30-120d), BUKAN sebagai bukti kuantitatif independen.

## Kesimpulan Langkah A

Sesuai dugaan rencana user: **RSF mentah (S(t) belum dikalibrasi) menghasilkan
median yang HAMPIR SELALU jauh lebih optimis dari kenyataan** - bukan sekadar
"kadang null", tapi ketika ADA angkanya, angkanya sistematis salah arah. Akar
penyebabnya konsisten dengan hipotesis user: kurva S(t) turun terlalu lambat
(under-predict risiko/over-predict survival), gap tumbuh dengan horizon.

`calibrators.joblib` (isotonic per horizon 30/60/90/120) sudah ada dan sudah
dipakai untuk `calibrated_risk_Nd` (4 titik diskrit, Fase R1a) - TAPI
`median_days_remaining`/`days_until_survival_90pct`/kurva yang ditampilkan
SEMUA masih dari kurva MENTAH (`predict/survival.py:218,237,248,310`), bukan
dari kurva yang sudah dikalibrasi - persis inkonsistensi yang diperingatkan
user di Langkah B rencana. Ini bukti kuat bahwa Langkah B (bangun kurva
terkalibrasi yang konsisten di seluruh grid, bukan cuma 4 titik) kemungkinan
besar berdampak nyata pada bias ini - diuji di laporan terpisah
(`rsf_median_curve_calibration_result.md`).
