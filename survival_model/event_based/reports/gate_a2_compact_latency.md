# Fase A2 lanjutan: latency kandidat compact (G6)

Model: kandidat compact A2 (n_estimators=50,
min_samples_leaf=100, grid dikasarkan).
Diukur pada `x_val` (fitur VALIDATION landmark, sudah di-encode) - proxy realistis
untuk bentuk data yang dilihat `predict.py` (jumlah kolom fitur sama).

| Metrik | Nilai | Ambang G6 (dari baseline_performance_catboost.md) | Status |
|---|---|---|---|
| Ukuran artifact | 66.2 MB | <=100 MB (keras) | LULUS |
| Cold load | 0.174 s | <=5 s | LULUS |
| Single predict p50 | 2.7 ms | <=3467,7 ms (1,5x baseline) | LULUS |
| Batch (ekstrapolasi 16.877 PART) | 2.7 s | <=94,0 s (2x baseline) | LULUS |

Batch di atas EKSTRAPOLASI linier dari chunk 2,000 baris (0.160 ms/baris) -
BUKAN pengukuran end-to-end lewat pipeline fitur production (itu perlu serving code
Fase C yang sesungguhnya, termasuk pembacaan database dan pembangunan fitur - di luar
cakupan studi kelayakan ini).
