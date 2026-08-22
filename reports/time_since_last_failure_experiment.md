# Eksperimen: hari sejak kerusakan terakhir per item_type (DITOLAK)

Dijalankan 2026-08-22, sisa Prioritas 1 roadmap ("time since last similar
failure"). Jalur samping - tidak menyentuh `models/failure/`.

## Hipotesis

Beda dari density rate 90/180d (JUMLAH kejadian dalam jendela), sinyal
recency ("berapa hari sejak kejadian TERAKHIR di item_type yang sama")
mungkin menangkap pola cluster/burst kerusakan yang tidak tertangkap rate.

## Metodologi

`time_since_last_group_failure()`: hari sejak `failure_onset_on` TERAKHIR
pada `item_type_at_install` yang sama, STRICT sebelum `observation_on`
(anti-leakage, `searchsorted(..., side="left")`). NaN (belum pernah ada
kejadian) -> 0 + flag `has_recent_item_type_failure`. Cakupan sinyal 92%
(bukan masalah data sparse).

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v4 (32 fitur) | 0,8319 | **0,1961** | 0,0210 | 0,3392 | 0,2175 |
| + time_since_last_item_type_failure (34) | 0,8331 | **0,1656** | 0,0213 | 0,3392 | 0,2175 |

PR-AUC turun TAJAM (-0,03, jauh lebih besar dari eksperimen fitur
sebelumnya yang turun -0,005 s/d -0,009). Recall/Presisi@kapasitas kali ini
TIDAK ikut naik (persis sama dengan baseline) - beda dari pola 3 eksperimen
sebelumnya (local density client+place, survival_risk_30d, interaksi) yang
setidaknya menunjukkan trade-off. Di sini murni kalah, tanpa kompensasi.

## Kesimpulan

**TIDAK di-wire.** Bukan masalah cakupan data (92% terisi) - sinyal recency
per kategori LUAS (item_type, cuma 5-6 kategori) sepertinya terlalu kasar/
noisy dibanding sinyal rate yang sudah ada, dan justru mengaburkan ranking
dibanding membantunya.

## Kalau mau dicoba lagi

Coba grouping yang lebih SEMPIT (item_model_code_clean, bukan item_type) -
"kapan model PART SPESIFIK ini terakhir rusak di armada" mungkin sinyal
lebih tajam daripada "kapan kategori LUAS ini terakhir rusak" (yang hampir
selalu punya jawaban "baru-baru ini" karena kategorinya besar, sehingga
variasi sinyalnya kecil/kurang informatif - konsisten dengan median 8,5
hari yang sangat pendek, wajar untuk grup sebesar item_type).
