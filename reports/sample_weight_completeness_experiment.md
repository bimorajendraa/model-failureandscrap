# Eksperimen: sample weight by completeness DITOLAK

Dijalankan 2026-08-22, roadmap "cara pakai data". Jalur samping.

## Hipotesis

Beda dari era-weighting (waktu, sudah ditolak - `reports/class_imbalance_experiment.md`):
baris dengan riwayat lebih KAYA (bukan lebih baru) diutamakan saat training,
lewat `completeness_score` (has_previous_cycle + log_total_prior_events
dinormalkan) sebagai `sample_weight` CatBoost - bukan sebagai fitur.

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline (tanpa weight) | 0,8319 | **0,1961** | 0,0210 | 0,3392 | 0,2175 |
| Weight linear (1,0-2,0) | 0,8352 | 0,1878 | 0,0211 | 0,3437 | 0,2203 |
| Weight halus (0,7-1,3) | 0,8338 | 0,1915 | 0,0211 | 0,3404 | 0,2182 |

Kedua rentang bobot gagal gerbang PR-AUC, walau Recall/Presisi@kapasitas
naik di keduanya (rentang lebih halus = trade-off lebih kecil, tapi tetap
gagal).

## Pola berulang - catatan lintas eksperimen

Ini kali KEENAM pola identik muncul di seri eksperimen roadmap ini
(local density client+place, survival-as-feature, 3 interaksi, sekarang
sample-weight-completeness): ROC-AUC & Recall/Presisi@kapasitas naik,
PR-AUC turun - konsisten cukup untuk disimpulkan ini BUKAN kebetulan
per-percobaan, tapi properti struktural populasi/metrik v4 saat ini:
hampir semua "penajaman sinyal tambahan" (fitur baru, reweighting) menggeser
ranking mendekati titik potong kapasitas dengan biaya di bagian lain kurva
precision-recall. Pengecualian satu-satunya yang benar-benar lolos gerbang
sejauh ini: local density item_type (naik di SEMUA metrik sekaligus,
lihat commit 385e4db) - kemungkinan karena itu sinyal genuinely baru
(bukan reweighting/kombinasi dari yang sudah ada), bukan sekadar
menekankan ulang sinyal yang sudah dipelajari model.

## Kesimpulan

**TIDAK di-wire.** Konsisten dengan seluruh seri eksperimen "cara pakai
data ulang" (era-weight, sample-weight-completeness) - keduanya kalah di
PR-AUC. Sinyal genuinely baru (seperti item_type density) tampaknya jauh
lebih mungkin lolos gerbang dibanding menekankan ulang bobot pada data yang
sudah ada.
