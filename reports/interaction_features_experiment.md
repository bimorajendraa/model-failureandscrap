# Eksperimen: interaksi eksplisit di CatBoost (DITOLAK)

Dijalankan 2026-08-22, sisa Prioritas 1 roadmap. Jalur samping - tidak
menulis ke `models/failure/`.

## Hipotesis

CatBoost bisa menangkap interaksi sendiri lewat kombinasi split pohon,
tapi eksplisit kadang membantu di data jarang-positif (~1,5% base rate) -
tiga interaksi dicoba: `age x prior_failures`, `trend_ratio x age`,
`item_type_failure_rate_90d x age`. Semuanya dibangun dari kolom yang
SUDAH ada di fitur v4 (perkalian langsung), tidak perlu query/join baru.

## Hasil (incremental di atas baseline v4, 32 fitur)

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v4 | 0,8319 | **0,1961** | 0,0210 | 0,3392 | 0,2175 |
| + age x prior_failures (33) | 0,8343 | 0,1871 | 0,0211 | 0,3404 | 0,2182 |
| + trend x age (34) | 0,8291 | 0,1892 | 0,0211 | 0,3415 | 0,2189 |
| + item_type_rate x age (35) | 0,8346 | 0,1908 | 0,0211 | 0,3426 | 0,2196 |

**Ketiganya gagal gerbang PR-AUC** (turun terus, tidak pernah pulih ke
baseline walau ditambahkan kumulatif), walau ROC-AUC dan Recall/Presisi@
kapasitas konsisten naik di setiap penambahan.

## Pola berulang lintas 3 eksperimen berturut-turut

Ini eksperimen KETIGA berturut-turut (setelah local density client+place,
survival_risk_30d) dengan pola identik: ROC-AUC & Recall/Presisi@kapasitas
naik, PR-AUC turun. Kemungkinan penjelasan: PR-AUC mengintegrasikan presisi
di SELURUH kurva recall, sementara Recall/Presisi@kapasitas cuma peduli
satu titik potong (kapasitas kerja tim). Fitur-fitur tambahan belakangan
ini tampaknya mempertajam urutan TEPAT di sekitar titik potong itu, dengan
biaya di bagian lain kurva - kemungkinan karena makin redundan/kolinear
dengan fitur density & degradasi yang sudah ada (v4 baseline SUDAH
memasukkan item_type density dan degradation features - ruang sinyal
"mudah" mungkin sudah banyak terpakai, sisa penambahan makin cenderung
tumpang tindih daripada menambah sinyal bersih baru).

## Kesimpulan

**TIDAK di-wire.** Pola berulang ini sendiri jadi catatan penting untuk
eksperimen fitur berikutnya di roadmap yang sama: kalau hasilnya lagi-lagi
"ROC-AUC/Recall@kapasitas naik, PR-AUC turun", ini mungkin bukan
kebetulan per-fitur, tapi tanda populasi fitur "mudah" untuk PR-AUC sudah
mendekati jenuh dengan v4 - worth dipertimbangkan sebelum terus menambah
fitur satu-satu dengan mekanisme serupa (interaksi/density/skor eksternal).
