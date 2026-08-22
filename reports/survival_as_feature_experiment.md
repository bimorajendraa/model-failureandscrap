# Eksperimen: survival_risk_30d (RSF) sebagai fitur CatBoost (DITOLAK)

Dijalankan 2026-08-22, Prioritas 2 roadmap. Jalur samping - tidak menulis ke
`models/failure/`, tidak membuat versi baru.

## Hipotesis

RSF event-based dan CatBoost "salah" di tempat berbeda (RSF menang PR-AUC,
kalah Recall/Presisi@kapasitas dibanding CatBoost saat dibandingkan
sendirian - lihat `survival_model/event_based/reports/gate_decision.md`).
Kalau skor RSF (`survival_risk_30d = 1 - S(30)`, dihitung pada
`observation_on` baris itu sendiri) dimasukkan sebagai FITUR TAMBAHAN ke
CatBoost, mungkin CatBoost bisa memanfaatkan sinyal itu tanpa mewarisi
kelemahan RSF.

## Metodologi

- Baseline = v4 production sungguhan (32 fitur), direproduksi persis.
- `training.landmark_eval.build_landmark_features_at_observation()` +
  `score_risk_30d_chunked()` (REUSE APA ADANYA - modul ini hasil Fase A1,
  scorer promosi permanen model survival, sudah menangani JEBAKAN #1 risk=
  1-S(30) langsung, JEBAKAN #2 support beku dari metadata, JEBAKAN #3
  chunking memori).
- **Bug ditemukan saat wiring**: `dataset` CatBoost SUDAH punya kolom
  turunannya sendiri (DEGRADATION_FEATURES/LOCAL_DENSITY_FEATURES/
  FLEET_FEATURES). `build_landmark_features_at_observation()` menghitung
  ulang beberapa nama yang SAMA lewat `attach_dynamic_extra()` yang
  memakai `pd.concat` (bukan assignment) untuk sebagian kolom -
  `ValueError: Cannot set a DataFrame with multiple columns` saat kolom
  duplikat itu coba dibaca `build_features()`. Diperbaiki dengan membuang
  ketiga daftar kolom itu dari `dataset` SEBELUM diserahkan ke
  `build_landmark_features_at_observation()` (fungsi itu menghitung
  semuanya sendiri dari awal, tidak butuh versi CatBoost).
- Diuji juga `log1p(survival_risk_30d)` - identik hasilnya dengan versi
  mentah (diharapkan: model tree-based tidak peka transformasi monoton,
  split threshold cuma bergeser posisi, keputusan split sama persis).

## Hasil

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v4 (32 fitur) | 0,8319 | **0,1961** | 0,0210 | 0,3392 | 0,2175 |
| + survival_risk_30d (34 fitur) | 0,8195 | 0,1910 | 0,0211 | **0,3437** | **0,2203** |
| + log1p(survival_risk_30d) (34 fitur) | 0,8195 | 0,1910 | 0,0211 | 0,3437 | 0,2203 |

**Trade-off yang bersih tapi gagal gerbang**: ROC-AUC dan PR-AUC turun
jelas, sementara Recall/Presisi@kapasitas justru naik. `decide_promotion`
mensyaratkan PR-AUC **dan** Recall@kapasitas **sama-sama** tidak boleh
turun - PR-AUC turun (0,1961->0,1910) berarti kandidat ini **gagal gerbang**
kalau benar-benar dilatih lewat jalur resmi, terlepas dari kenaikan
Recall@kapasitas.

## Analisis

RSF dan CatBoost memang "salah di tempat berbeda" seperti hipotesis awal -
buktinya persis pola trade-off yang sama terlihat di `gate_decision.md`
(RSF sendirian: PR-AUC menang, Recall/Presisi@kapasitas kalah). Tapi
menggabungkannya sebagai FITUR TAMBAHAN tidak otomatis mengambil "yang
terbaik dari keduanya" - CatBoost sepertinya justru SEBAGIAN mengganti
sinyalnya sendiri yang lebih tajam (untuk urutan keseluruhan) dengan sinyal
RSF yang lebih kasar, menghasilkan kompromi yang kalah di metrik yang
paling menentukan (PR-AUC) demi naik tipis di metrik lain.

## Kesimpulan

**TIDAK di-wire ke production sebagai fitur tunggal `survival_risk_30d`.**
Trade-off-nya nyata tapi tidak lolos aturan promosi yang sudah ada (PR-AUC
tidak boleh turun).

## Kalau mau dicoba lagi

- **Soft-voting/stacking** (menggabungkan SKOR AKHIR kedua model, bukan
  memasukkan skor RSF sebagai fitur mentah di tengah training CatBoost) -
  pendekatan berbeda dari yang dicoba di sini, belum diuji.
- Horizon RSF lain (60/90/120 hari) sebagai fitur tambahan - belum dicoba
  sesuai rencana awal ("baru setelah 30d terbukti berguna" - 30d sendiri
  belum terbukti, jadi ditunda).
- Kalau distribusi trade-off ini konsisten (PR-AUC turun, Recall@kapasitas
  naik) di percobaan lanjutan, mungkin ini sinyal bahwa KAPASITAS kerja
  tim (200/bulan) atau bobot metrik promosi perlu didiskusikan ulang -
  bukan cuma soal fitur mana yang ditambahkan.
