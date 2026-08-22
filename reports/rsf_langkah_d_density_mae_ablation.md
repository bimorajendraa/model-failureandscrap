# Langkah D — ablation item_type density di RSF, digate MAE median + kalibrasi

## Konteks

`reports/rsf_r2_item_type_density_ablation.md` sudah menolak fitur ini dengan
kriteria Brier/C-index (gate R3). Rencana user eksplisit meminta kriteria yang
berbeda untuk Langkah D: *"Promote fitur hanya jika MAE median / calibration
membaik di holdout, bukan jika C-index naik."* - laporan ini menguji ulang
kandidat yang SAMA dengan metodologi Langkah A/B (MAE median + gap kalibrasi),
bukan mengulang R2 apa adanya.

## Metodologi

Sama persis dengan R2: baseline vs baseline+4 kolom `LOCAL_DENSITY_FEATURES`
(item_type 90/180d), encoder kategorikal identik, RSF compact params production
untuk kedua varian. BEDA dari R2: kedua model di sini JUGA dikalibrasi penuh
lewat `curves.calibrate_curve()` (Langkah B) dan dievaluasi dengan MAE median
(baris `event_observed=True` & median terisi) + gap kalibrasi 30d/90d - metodologi
identik dengan `rsf_median_curve_baseline.md`/`rsf_median_curve_calibration_result.md`.

## Hasil

| Metrik | Baseline | Candidate (+density) | Verdict |
|---|---:|---:|---|
| MAE median, seluruh TEST (n_usable=386/380) | 450,0 hari | 480,0 hari | **MEMBURUK** (+6,7%) |
| MAE median, ANCHOR (n_usable=27/25) | 609,7 hari | 446,5 hari | membaik (lihat catatan) |
| % null, seluruh TEST | 60,4% | 61,6% | ~tidak berubah |
| % null, ANCHOR | 75,0% | 73,9% | ~tidak berubah |
| Gap kalibrasi 30d (n_usable=4.382) | 0,0296 | 0,0312 | **MEMBURUK** |
| Gap kalibrasi 90d (n_usable=2.430) | 0,0539 | 0,0547 | **MEMBURUK** |

**Catatan penting soal ANCHOR MAE**: n_usable di subset ini cuma 25-27 baris -
jauh lebih kecil dan lebih rawan noise dibanding seluruh TEST (n≈380-386).
Perbaikan 609,7->446,5 hari mungkin cuma pergeseran beberapa PART spesifik,
bukan pola sistematis - TIDAK dijadikan dasar keputusan sendirian.

## Verdict: **DITOLAK** (konsisten dengan R2, kriteria berbeda - kesimpulan sama)

3 dari 4 metrik yang lebih bisa dipercaya (MAE seluruh TEST + gap kalibrasi
30d + gap kalibrasi 90d, ketiganya dengan sampel jauh lebih besar) MEMBURUK.
Satu-satunya perbaikan (ANCHOR MAE) bertumpu pada sampel terlalu kecil untuk
dipercaya sendirian. Dua metodologi evaluasi BERBEDA (R2: Brier/C-index; sini:
MAE median/kalibrasi) menghasilkan kesimpulan YANG SAMA - item_type density
tidak membantu RSF, baik untuk ranking maupun untuk ketepatan waktu. Tidak ada
perubahan kode production dari eksperimen ini.

## Langkah E — resolusi grid (cek, bukan eksperimen)

Diperiksa lewat inspeksi metadata (bukan training baru): grid waktu RSF
(`times_grid`) resolusi HARIAN sampai 120 hari, lalu langkah 60 harian
setelahnya (`coarsen_duration_days()`, hasil Fase A2 - artifact 5,26 GB -> 66,2
MB tanpa kompromi C-index VALIDATION). `CURVE_STEP_DAYS=30` (sampling titik
GRAFIK yang ditampilkan) tidak membatasi PERHITUNGAN median/p90/ambang -
`median_survival_time()` dkk selalu membaca grid ASLI (harian ≤120 hari), jadi
angka yang dilaporkan API TIDAK terpengaruh oleh resolusi tampilan grafik.

**Temuan nyata**: zona 120-180 hari HANYA punya SATU titik tambahan (t=180) -
resolusi di situ kasar (60 hari), padahal Langkah A/B menemukan mayoritas
median (saat terisi) jatuh di ratusan hari, sebagian di zona ini. Memperhalus
resolusi 120-365 hari (mis. jadi 30-harian) BUTUH retrain dengan skema
coarsening berbeda - trade-off LANGSUNG dengan ukuran artifact (yang sengaja
dikecilkan di Fase A2 dari 5,26 GB) dan kecepatan serving. Ini keputusan
arsitektur yang sengaja TIDAK diambil sepihak di sini - dilaporkan sebagai
temuan untuk dipertimbangkan, bukan dieksekusi tanpa persetujuan.
