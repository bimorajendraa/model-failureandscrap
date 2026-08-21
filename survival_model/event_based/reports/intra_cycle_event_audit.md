# Audit: kekayaan event di dalam siklus + ketersediaan device/usage context

Dua audit data mentah yang menentukan desain `eb_src/landmark_builder.py`
dan menjawab Tahap 2/12 (device context, usage intensity) - dikerjakan
SEBELUM menulis kode landmark, bukan sesudah.

## 1. Event operasional di dalam satu installation cycle

Pertanyaan: kalau landmark dibuat "hanya saat informasi berubah" (event
organik), berapa banyak titik observasi yang realistis dihasilkan per
lifecycle?

Metodologi: untuk setiap 23.927 lifecycle cohort (`is_initial_model_cohort`,
sama dengan populasi model statis), hitung event operasional (`data_reader.
get_events()`) yang timestamp-nya STRICTLY di antara `installed_on` dan
`cycle_end_on` lifecycle itu (exclude event batas cycle itu sendiri).

**Hasil:**

| | Jumlah lifecycle | Persentase |
|---|---:|---:|
| 0 event di tengah siklus | 19.208 | 80,3% |
| >=1 event di tengah siklus | 4.719 | 19,7% |

Di antara yang punya event (4.719 lifecycle): rata-rata 2,03 event, median
1, maksimum 16.

Sampel event yang muncul (pola dominan): `REPAIRED -> CORRECTIVE REQUESTED
-> ISSUED -> DELIVERY` - logistik repair siklus SEBELUMNYA yang timestamp-nya
kebetulan jatuh sebelum `INSTALLED` event yang membuka siklus BERIKUTNYA
tercatat secara berurutan wajar (bukan anomali per-item, tapi juga bukan
"event di tengah siklus AKTIF" dalam arti operasional - part sedang dalam
proses reinstall, bukan terpasang & dipantau).

**Kesimpulan**: skema "observation hanya saat event organik" (Tahap 7)
SENDIRIAN tidak cukup - 80% lifecycle hanya akan punya SATU observasi
(install), sama seperti model statis. Anchor jarang (90/180/365 hari, lalu
+365) jadi sumber UTAMA landmark tambahan untuk lifecycle yang bertahan
lama, bukan pelengkap opsional seperti tersirat di spesifikasi awal - ini
konsisten dengan instruksi asli ("90-day sparse anchor... optional jika
tidak ada event lama"), hanya saja kasus "tidak ada event lama" ternyata
adalah kasus MAYORITAS (80%), bukan minoritas.

## 2. Device type / device model / usage intensity (Tahap 2 & 12)

`survival_model/README.md` (sesi sebelumnya) menyimpulkan `device_type`/
`device_model` "tidak tersedia lewat relasi yang sudah dikanonikalisasi" -
kesimpulan itu diverifikasi ULANG di sini lewat sweep skema database penuh
(`information_schema.columns ILIKE '%device%'`), BUKAN dipercaya begitu
saja, sesuai instruksi eksplisit ("terutama pastikan device_type dan
device_model benar-benar masuk eksperimen jika tersedia").

**Ditemukan**: `journal.replacement_history` - tabel yang TIDAK direferensikan
sama sekali oleh `data_reader.py`/`feature_builder.py`/`survival_model/`
manapun. Kolom: `spare_part_serial_code`, `device_type`, `device_model_name`,
`device_serial_code`, `install_time`, `failure_time`, `failure_status`,
`total_hours`, `total_journey_per_part`.

**Kualitas & cakupan data (12.695 baris):**

| Pemeriksaan | Hasil |
|---|---|
| Rentang `install_time` | **2025-01-01 s/d 2026-08-03 - 0 baris sebelum 2025** |
| `device_type` (4 nilai) | GATE (10.680), CVIM (1.431), BALANCE READER (331), POS (253) |
| `total_hours` | min=-2020, max=14293, mean=10005 (ADA nilai NEGATIF) |
| Anomali timestamp | Ditemukan baris dengan `install_time` > `failure_time` (mis. install 2025-08-21, failure 2025-05-29 - install SETELAH failure) |
| Skema identifier `spare_part_serial_code` | Campuran format pairing-code (`XXXXXXX-XXXXXXXXXXXXXXX-XX`) DAN host-code pendek (`XXXXXXXXXXXXXXX`) - butuh dual-lookup baru seperti `_matches_inventory()`, bukan join langsung |

**Kesimpulan: DITOLAK, bukan diam-diam dilewati.** Alasan utama definitif:
TRAIN split (`installed_on < validation_start`, mencakup 2014-2024) akan
punya **0% coverage** pada tabel ini (baris paling awal 2025-01-01) - model
tidak mungkin belajar dari kolom yang 100% kosong di SELURUH data latih.
Ini murni keterbatasan cakupan temporal tabel sumbernya sendiri (kemungkinan
sistem/tabel baru, belum ada backfill historis), BUKAN sesuatu yang bisa
diperbaiki lewat rekayasa fitur. Anomali kualitas data (total_hours negatif,
timestamp terbalik) dan skema identifier campuran adalah alasan SEKUNDER -
bahkan andai cakupannya penuh, data ini masih butuh pembersihan signifikan
sebelum dipakai.

Sesuai Tahap 13: ini konkret salah satu contoh "faktor penting yang tidak
tersedia di database" (usage intensity/operating hours) - **bukan** tidak
ada sama sekali di database, tapi ada dalam bentuk yang belum bisa
dipertanggungjawabkan untuk dipakai (cakupan temporal, kualitas data).
Kalau tabel ini di-backfill historis di masa depan (2014-2024) dan
anomalinya dibersihkan, ini kandidat kuat untuk diaudit ulang - bukan
ditutup permanen.
