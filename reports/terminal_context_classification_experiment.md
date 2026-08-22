# Eksperimen: terminal/device context di model kerusakan CatBoost (DITOLAK)

Dijalankan 2026-08-22, Prioritas 1.1 dari roadmap peningkatan akurasi. Jalur
samping - tidak menulis ke `models/failure/`, tidak membuat versi baru.

## Hipotesis

`terminal_type_grouped` (device tempat PART dipasang: GATE/CVIM/POS/BALANCE
READER/VENDING MACHINE) sudah terbukti +0,007-0,019 C-index di landmark
survival event-based. Pertanyaannya: apakah sinyal yang sama berguna untuk
model kerusakan CatBoost (klasifikasi biner 30-hari)?

## Metodologi

- Baseline = v3 production sungguhan (28 fitur, hasil `build_dataset()`
  asli) - direproduksi di harness dan dicocokkan ke metadata v3 (ROC-AUC/
  PR-AUC/Brier/Recall&Presisi@kapasitas SAMA PERSIS dengan
  `models/failure/v3/metadata.json`).
- `data_reader.get_terminal_context()` (fleet-wide) + `features.survival.
  terminal_context.attach_terminal_context()` (REUSE apa adanya - join
  point-in-time safe, hanya `VALID_POINT_IN_TIME_RELATION` yang dipakai,
  sisanya UNKNOWN).
- Grouping via `features.survival.categorical_support` (REUSE apa adanya),
  threshold=100 - dicek dulu distribusinya: cuma 5 kategori device,
  SEMUANYA bersupport tinggi (minimum 6.451 baris) - threshold LOW_SUPPORT
  praktis tidak berpengaruh di sini, bukan sumber masalah.
- VALIDATION/TEST TIDAK disentuh selain fitur tambahan ini.

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v3 (28 fitur) | 0,8244 | 0,1884 | 0,0211 | 0,3392 | 0,2175 |
| + terminal_type_grouped (29 fitur) | 0,8221 | 0,1707 | 0,0213 | 0,3271 | 0,2097 |

**Kalah di kelima metrik sekaligus.**

Distribusi `terminal_type_grouped` di populasi TRAIN+VAL+TEST (356.100 baris):
UNKNOWN 142.783 (40,1%), GATE 101.187, POS 68.751, CVIM 29.947,
VENDING MACHINE 6.783, BALANCE READER 6.451, LOW_SUPPORT 198.

## Analisis kenapa gagal (berbeda dari survival)

1. **Cakupan cuma 59,9%** - 40% baris dapat UNKNOWN (relasi device baru
   "diketahui" setelah instalasi, wajib dibuang sesuai aturan point-in-time
   `parent_link_quality_status`). Bukan penyebab utama sendirian (survival
   punya keterbatasan cakupan serupa dan tetap untung), tapi berkontribusi.
2. **Beda horizon yang dijawab.** Survival diukur pakai C-index - kualitas
   ranking waktu-ke-kejadian lintas horizon PANJANG (bulan-tahun). Model
   kerusakan CatBoost di sini menjawab jendela SEMPIT (30 hari). Device
   tempat PART dipasang kemungkinan berkorelasi dengan pola degradasi
   JANGKA PANJANG (beban kerja kumulatif, lingkungan operasional) - sinyal
   yang relevan untuk "kapan akhirnya rusak" tapi tidak cukup informatif
   untuk "apakah rusak 30 hari dari sekarang khususnya".
3. CatBoost kemungkinan "membayar" kompleksitas pohon tambahan untuk fitur
   yang informatifnya rendah di jendela sempit, sedikit mengorbankan
   ketajaman split di fitur lain yang lebih relevan - konsisten dengan
   Brier yang juga sedikit memburuk (kalibrasi ikut terganggu, bukan cuma
   ranking).

## Kesimpulan

**TIDAK di-wire ke production.** Sinyal terminal/device context terbukti
BERGUNA untuk pertanyaan survival ("berapa lama lagi"), TIDAK terbukti
berguna untuk pertanyaan classification 30-hari ("apakah rusak bulan
depan") - dua pertanyaan yang secara struktural berbeda cukup untuk
membuat fitur yang sama tidak otomatis transfer. Pelajaran: keberhasilan
fitur di satu model TIDAK bisa diasumsikan transfer ke model lain walau
keduanya memprediksi hal yang berkaitan (kerusakan PART) - selalu perlu
ablation terpisah per model, per metrik yang benar-benar dipakai
(Recall/Presisi@kapasitas classification), bukan cuma metrik model asalnya
(C-index survival).

Kalau pertanyaan ini muncul lagi: jangan ulang tanpa bukti baru (mis. data
cakupan device membaik signifikan, atau horizon target classification
berubah dari 30 hari).
