# Eksperimen: grid hyperparameter CatBoost - baseline tetap terbaik

Dijalankan 2026-08-22, roadmap "hyperparameter & training protocol". Jalur
samping - tidak menyentuh `models/failure/`.

## Metodologi

5 kandidat (baseline + 4 varian lebih dalam/lebih banyak iterasi), dilatih
di atas fitur v4 (32 fitur). **Seleksi via VAL PR-AUC** (bukan TEST - TEST
cuma dilaporkan untuk kandidat terpilih, disiplin yang sama seperti
`reports/category_threshold.md`).

## Hasil

| Config | VAL PR-AUC | VAL ROC-AUC |
|---|---:|---:|
| **Baseline (depth=4, iterations=200, lr=0,03, l2=10)** | **0,1116** | 0,8170 |
| depth=5, iterations=300, l2=12 | 0,1098 | 0,8124 |
| depth=6, iterations=400, l2=15 | 0,1065 | 0,8089 |
| depth=5, iterations=300, lr=0,025, l2=12 | 0,1098 | 0,8173 |
| depth=4, iterations=300, l2=15 | 0,1087 | 0,8155 |

**Baseline menang di VAL PR-AUC melawan SEMUA 4 kandidat lain** - tidak ada
yang perlu dievaluasi di TEST karena baseline sendiri yang terpilih.

## Kesimpulan

**Konfigurasi sekarang (depth=4, iterations=200) sudah dekat optimal**
untuk grid yang dicoba. Tree lebih dalam/lebih banyak iterasi konsisten
LEBIH BURUK di VAL - masuk akal untuk base rate positif ~1,5%: kompleksitas
tambahan kemungkinan overfit ke noise, bukan menangkap sinyal baru.
Tidak ada perubahan hyperparameter yang direkomendasikan dari grid ini.

## Kalau mau dicoba lagi

- Grid ke arah SEBALIKNYA (lebih dangkal/regularisasi lebih kuat, depth=3,
  l2 lebih besar) belum dicoba - kalau depth lebih dalam konsisten kalah,
  mungkin optimal ada di sisi yang lebih sederhana, bukan lebih kompleks.
- `auto_class_weights` sudah terbukti perlu (reports/class_imbalance_experiment.md)
  - grid ini tidak mengubah itu, cuma depth/iterations/l2/learning_rate.
