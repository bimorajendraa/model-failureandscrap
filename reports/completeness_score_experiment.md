# Eksperimen: completeness/data-quality score DITOLAK

Dijalankan 2026-08-22, item terakhir sisa Prioritas 1 roadmap. Jalur samping.

## Hipotesis

Data lama (~2013-2017) diduga kurang lengkap pencatatannya. Dicoba sebagai
FITUR (bukan sample weight - era-weighting sudah dicoba & ditolak,
`reports/class_imbalance_experiment.md`) supaya model bisa belajar
"kurang yakin" pada baris dari era itu, alih-alih dibobot ulang saat training.

Dua varian:
1. `is_early_era`: `observation_on` tahun < 2018.
2. `completeness_score`: gabungan `has_previous_cycle` + `log_total_prior_events`
   dinormalkan - proxy "riwayat tipis vs lengkap".

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v4 (32 fitur) | 0,8319 | 0,1961 | 0,0210 | 0,3392 | 0,2175 |
| + is_early_era (33) | 0,8327 | 0,1767 | 0,0212 | 0,3337 | 0,2139 |
| + completeness_score (34) | 0,8356 | 0,1870 | 0,0211 | 0,3370 | 0,2161 |

**Kedua varian kalah di SEMUA metrik operasional** (PR-AUC, Recall@kapasitas,
Presisi@kapasitas) - cuma ROC-AUC yang naik. Beda dari eksperimen lain di
seri ini (yang setidaknya Recall@kapasitas ikut naik sebagai kompensasi) -
di sini murni kalah.

## Kesimpulan

**TIDAK di-wire.** Distribusi tahun observasi (2013: 7.771 baris s/d 2026:
38.451 baris, makin baru makin banyak) menunjukkan data memang tidak
seimbang antar era, tapi menandai era secara eksplisit sebagai fitur
sepertinya membuat model "membedakan berdasarkan waktu" alih-alih
"membedakan berdasarkan kondisi PART" - kontraproduktif untuk generalisasi.
