# Error analysis + eksperimen terarah: age_history_base_rate (DITOLAK)

Dijalankan 2026-08-22, roadmap "error analysis -> fitur terarah". Jalur
samping - tidak menyentuh `models/failure/`.

## Error analysis (model v4 SUNGGUHAN, bukan refit, di TEST split)

902 kerusakan nyata di TEST, model v4 menangkap 306 (Recall@kapasitas
0,339) dari 1.407 slot. Breakdown per slice:

| Slice | Recall |
|---|---:|
| Umur pasang 0-90 hari | 0,812 |
| Umur pasang 91-180 hari | 0,075 |
| Umur pasang 181-365 hari | 0,000 |
| Umur pasang 366-730 hari | 0,000 |
| Umur pasang 731-1460 hari | 0,000 |
| Umur pasang 1461+ hari | 0,000 |
| Tanpa riwayat corrective sebelumnya | 0,000 (347 kerusakan, TIDAK SATU PUN tertangkap) |
| Ada riwayat corrective sebelumnya | 0,551 |
| First failure | 0,197 |
| Repeat failure | 0,527 |
| Client KAI LRT Jabodebek | 0,037 |
| Client KCI (commuter) | 0,369 |

**Temuan struktural jelas**: model nyaris buta terhadap PART berumur >90
hari TANPA riwayat corrective sebelumnya (38% dari total positif TEST).
Masuk akal secara arsitektur - hampir semua fitur utama model (count-based:
`log_prior_failure_count`, `log_prior_corrective_count`, density item_type,
dst) bernilai NOL untuk populasi ini, satu-satunya pembeda yang tersisa
adalah identitas+umur, dan model belum menemukan pola berguna di situ.

Umur 0-90 hari justru BAIK (0,812) - kemungkinan kegagalan dini
(infant mortality/cacat produksi) berkorelasi kuat dengan
`part_model_category`/batch, tetap bisa ditebak dari identitas walau
tanpa riwayat individual.

## Eksperimen terarah: age_history_base_rate

**Hipotesis**: smoothed target encoding (shrinkage ke rata-rata global,
`SMOOTH_STRENGTH=50`) base rate kerusakan 30-hari historis per kombinasi
(item_type_at_install, installation_age_band, has_prior_corrective),
dihitung dari TRAIN saja dan DIBEKUKAN (pola sama `part_model_support`) -
memberi CatBoost sinyal populasi langsung untuk kelompok yang tidak
punya riwayat individual.

### Hasil

| Metrik | Baseline v4 | + age_history_base_rate |
|---|---:|---:|
| ROC-AUC | 0,8319 | 0,8109 |
| PR-AUC | 0,1961 | 0,1788 |
| Brier | 0,0210 | 0,0212 |
| Recall@kapasitas | 0,3392 | 0,3082 |
| Presisi@kapasitas | 0,2175 | 0,1976 |

**Kalah di SEMUA metrik, tanpa trade-off** (beda dari kebanyakan eksperimen
lain di seri ini yang setidaknya menaikkan sesuatu).

### Cek langsung: apakah blind spot membaik?

| Slice | Recall v4 asli | Recall dengan fitur baru |
|---|---:|---:|
| Umur 181-365 hari | 0,000 | 0,015 |
| Umur 366-730 hari | 0,000 | 0,000 |
| Umur 731-1460 hari | 0,000 | 0,000 |
| Umur 1461+ hari | 0,000 | 0,000 |
| Tanpa riwayat corrective | 0,000 | **0,000** |
| Umur 0-90 hari (tadinya bagus) | 0,812 | 0,734 (ikut turun) |

**Gagal total mencapai tujuannya** - blind spot tetap ~0%, dan kelompok
yang tadinya bagus (0-90 hari) malah ikut memburuk.

## Analisis kenapa gagal

180 kombinasi bucket dari ~250rb baris TRAIN dengan base rate cuma 1,5%
berarti rata-rata ~21 kejadian positif per bucket - terlalu sedikit untuk
target-encoding yang stabil walau sudah di-smooth. Sinyalnya lebih banyak
noise daripada informasi, dan tampaknya mengalihkan CatBoost dari split
yang sudah bekerja baik di tempat lain (populasi 0-90 hari).

## Kesimpulan

**TIDAK di-wire.** Tapi temuan ini SENDIRI penting: ini eksperimen PALING
TERARAH di seluruh seri (langsung dari error analysis, bukan tebakan) dan
tetap gagal memperbaiki populasi "PART lama tanpa riwayat". Bukti kuat
bahwa blind spot ini kemungkinan besar **keterbatasan data**, bukan fitur
yang hilang - PART yang gagal tanpa peringatan sebelumnya mungkin butuh
sinyal yang genuinely tidak ada di dataset ini (intensitas pemakaian,
kondisi lingkungan, kualitas batch produksi - lihat README soal
keterbatasan data serupa untuk device_type/device_model).

## Rekomendasi ke depan

Populasi "no prior corrective, umur lama" (38% dari kerusakan TEST)
kemungkinan besar TIDAK bisa diprediksi lebih baik dengan data yang ada
sekarang. Kalau mau ditingkatkan, opsi realistisnya BUKAN fitur baru dari
data yang sama, tapi:
1. Data baru yang belum ada (intensitas pemakaian, environmental sensor,
   batch produksi) - di luar cakupan proyek ini.
2. Terima keterbatasan ini secara eksplisit di operasional (mis. jadwal
   inspeksi preventif berbasis umur murni untuk kelompok ini, terpisah
   dari model prediktif - bukan soal akurasi model, tapi kebijakan
   operasional).
