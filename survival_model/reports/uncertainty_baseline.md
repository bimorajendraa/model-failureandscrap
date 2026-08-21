# Ketidakpastian baseline C-index (sebelum eksperimen model-family/fitur baru)

VALIDATION: 2316 baris, 385 event - kecil, jadi C-index titik tunggal bisa menyesatkan. Dua sumber ketidakpastian diukur terpisah: (1) bootstrap resampling baris VAL pada model produksi SAAT INI (model tidak berubah, hanya baris mana yang masuk perhitungan C-index yang berubah), (2) variasi antar random_state RSF (model berubah, baris VAL tetap). Kandidat model/fitur baru pada langkah berikutnya (model_family.md dst.) HANYA dianggap menang kalau VAL C-index-nya di LUAR rentang berikut, bukan menang tipis di dalam noise ini.

## Bootstrap CI (200 resample, model produksi saat ini)
| Model | Point estimate | 95% CI lower | 95% CI upper | Std |
|---|---|---|---|---|
| random_survival_forest | 0.8114 | 0.7926 | 0.8289 | 0.0096 |
| cox_ph | 0.7819 | 0.7603 | 0.8011 | 0.0103 |

## Variasi antar seed RSF (5 seed, hyperparameter & fitur sama)
| Seed | VAL C-index |
|---|---|
| 0 | 0.8115 |
| 1 | 0.8086 |
| 2 | 0.8111 |
| 3 | 0.8120 |
| 4 | 0.8106 |

Rentang antar-seed: [0.8086, 0.8120] (std=0.0012).