# Eksperimen: local failure density (item_type/client/place) di CatBoost

Dijalankan 2026-08-22, Prioritas 1.2 roadmap. Jalur samping sampai bagian
"Kesimpulan" - item_type density kemudian di-wire ke production (lihat
commit terpisah), client/place TIDAK.

## Hipotesis

Generalisasi `attach_fleet()` (yang sudah ada, dikelompokkan per
`item_model_code_clean`) ke tiga dimensi lain: laju kerusakan point-in-time
per **item_type_at_install** (kategori lebih luas), **client**, dan
**place/lokasi** dalam jendela 90/180 hari.

## Metodologi

- Baseline = v3 production sungguhan (28 fitur), direproduksi persis
  (metrik cocok sampai 4 desimal dengan `models/failure/v3/metadata.json`).
- `local_density(observations, cycles, failures, group_column, window_days)`:
  generalisasi PERSIS mekanisme `attach_fleet()` (reuse `_count_before`),
  cuma `group_column` diparameterisasi.
- **item_type_at_install**: tersedia LANGSUNG di kedua sisi tanpa join baru
  - `episodes["item_type_clean"]` (dari `get_failure_episodes()`) dan
  `cycles` + `install_context.attach_install_context()` (REUSE join
  point-in-time yang sudah dipakai survival). Cakupan 100%.
- **client/place**: `episodes` TIDAK punya kolom client/place langsung -
  perlu range-join ke cycle asalnya (`installed_on <= failure_onset_on <=
  cycle_end_on`, per item). Diverifikasi: match rate cuma **5.897/6.715
  (87,8%)** - 818 kejadian tidak match cycle manapun untuk item itu (bukan
  `is_initial_model_cohort`, atau data quirk lain) - tidak dipaksakan,
  murni tidak ikut menyumbang hitungan.

## Hasil (incremental di atas baseline v3)

| Varian | ROC-AUC | PR-AUC | Brier | Recall@kap | Presisi@kap |
|---|---:|---:|---:|---:|---:|
| Baseline v3 (28 fitur) | 0,8244 | 0,1884 | 0,0211 | 0,3392 | 0,2175 |
| + item_type density (32) | 0,8319 | **0,1961** | 0,0210 | 0,3392 | 0,2175 |
| + item_type + client (36) | 0,8316 | 0,1705 | 0,0213 | 0,3370 | 0,2161 |
| + item_type + client + place (40) | 0,8322 | 0,1860 | 0,0212 | 0,3459 | 0,2217 |

## Kesimpulan per dimensi

**item_type: MENANG BERSIH** - naik di kelima metrik, tanpa trade-off.
Di-wire ke production (lihat commit terpisah "Fase ... item_type density").

**client: MERUGIKAN** - PR-AUC/Brier/Recall/Presisi semua turun begitu
ditambahkan di atas item_type. Kemungkinan besar karena cakupan join
87,8% - 12% kejadian yang hilang cukup untuk membuat sinyalnya berisik,
bukan membantu. **TIDAK di-wire.**

**place: menambal sebagian kerusakan client** (Recall/Presisi@kapasitas
malah jadi TERTINGGI dari semua varian pada kombinasi penuh), tapi PR-AUC
gabungan (0,1860) tetap DI BAWAH baseline v3 murni (0,1884). Karena
`decide_promotion` mensyaratkan PR-AUC **dan** Recall@kapasitas sama-sama
tidak boleh turun, kombinasi client+place **gagal gerbang** dibanding v3
saat ini - trade-off yang tidak lolos aturan promosi yang ada.
**TIDAK di-wire dalam bentuk sekarang.**

## Kalau mau dicoba lagi

- **place SENDIRIAN** (tanpa client) belum diuji terpisah - mungkin ada
  nilai, belum dibuktikan.
- Perbaiki join client/place terlebih dulu (cari cara mendapat cakupan
  >87,8%, mis. matching yang lebih longgar atau sumber data client/place
  yang berbeda) sebelum mengulang eksperimen ini - jangan ulang dengan
  join yang sama tanpa perbaikan, hasilnya kemungkinan besar sama.
