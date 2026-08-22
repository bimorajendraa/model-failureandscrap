# Eksperimen: repair-quality proxy (confirmed-failure previous cycle) DITOLAK

Dijalankan 2026-08-22, sisa Prioritas 1 roadmap ("repair-quality proxy" -
"gagal lagi cepat setelah repair"). Jalur samping.

## Hipotesis

`previous_cycle_lifetime_mean` yang SUDAH ADA di model mencampur SEMUA cara
siklus sebelumnya berakhir (termasuk masih aktif/dipindah tanpa kerusakan) -
bukan "lifetime sampai gagal" seperti namanya menyiratkan (lihat
`reports/previous_cycle_audit.md`, ditemukan waktu membangun model
survival). Versi CONFIRMED-failure (`features/survival/previous_cycle.py`,
REUSE apa adanya) mengukur lebih jujur "seberapa cepat part serupa gagal
lagi setelah benar-benar rusak dan dipasang ulang" - sinyal repair-quality
yang lebih langsung.

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v4 (32 fitur) | 0,8319 | 0,1961 | 0,0210 | 0,3392 | 0,2175 |
| + confirmed_failure_lifetime_mean (34) | 0,8323 | 0,1951 | 0,0211 | 0,3392 | 0,2175 |
| + last_confirmed_failure_lifetime (36) | 0,8316 | 0,1871 | 0,0211 | 0,3381 | 0,2168 |

**Cakupan sinyal cuma 9,3%** (mayoritas PART belum pernah punya siklus
sebelumnya yang CONFIRMED berakhir kerusakan - masuk akal, kebanyakan masih
di siklus pertama atau siklus sebelumnya berakhir bukan karena rusak).

Varian rata-rata (confirmed_failure_lifetime_mean): praktis NETRAL - PR-AUC
turun 0,001 (level noise), Recall/Presisi@kapasitas persis sama. Varian
"terakhir" (last_confirmed_failure_lifetime, ditambahkan di atas rata-rata):
lebih buruk di semua metrik.

## Kesimpulan

**TIDAK di-wire.** Cakupan terlalu sparse (9,3%) untuk sinyal ini mengubah
model secara meaningful - bukan sinyal yang salah arah seperti eksperimen
sebelumnya, tapi juga tidak cukup data untuk benar-benar membantu. Sesuai
aturan `decide_promotion` (PR-AUC tidak boleh turun sama sekali, sekecil
apa pun), varian mean pun gagal gerbang secara ketat walau bedanya
sangat kecil.
