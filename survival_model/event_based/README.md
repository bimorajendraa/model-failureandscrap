# event_based — landmark/dynamic survival (Tahap 6-9)

Eksperimen lanjutan dari `survival_model/` (model statis, **tidak diubah**
oleh folder ini). Model statis mengobservasi tiap lifecycle **satu kali**,
pada `installed_on`. Modul ini mengobservasi tiap lifecycle **berkali-kali**
(landmark), dengan fitur riwayat/armada/degradasi dihitung ULANG pada tiap
titik - menjawab "dengan kondisi PART SAAT INI, berapa lama lagi sampai
failure?", bukan "saat pertama dipasang, berapa lama akan bertahan?".

## Hasil final (lihat reports/evaluation_report.md untuk detail lengkap)

| Metrik | Static (baseline) | Event-based FULL landmark | Event-based **t0-only** (adil) |
|---|---|---|---|
| VAL C-index (RSF) | 0,8114 | 0,8290 | **0,7985** |
| TEST C-index (RSF) | 0,8082 | 0,8477 | **0,8105** |
| PR-AUC operasional | 0,1633 | - | **0,1824** |
| ROC-AUC operasional | 0,6871 | - | 0,6961 |
| Recall@kapasitas | 0,3108 | - | **0,3401** |

**Kesimpulan: pada C-index murni (t0-only, perbandingan adil), event-based
masih sedikit di bawah static (0,7985 vs 0,8114) - tapi pada metrik
OPERASIONAL yang sebenarnya dipakai bisnis, event-based sudah LEBIH BAIK**
(PR-AUC +0,019, Recall@kapasitas +0,029). Angka "full landmark" (0,8290)
BUKAN perbandingan apples-to-apples - lihat "Kenapa ada 3 lapis evaluasi".

Perjalanan VAL t0-only: 0,7849 (fitur dasar) -> 0,7954 (+ degradation trend
+ cumulative usage + jendela corrective) -> **0,7985 (+ device/terminal
context, query kanonikal baru)**. Setiap kelompok fitur baru konsisten
positif, tidak ada yang menurunkan skor.

## Desain landmark

Audit data (`reports/intra_cycle_event_audit.md`) menemukan **80,3% dari
23.927 lifecycle TIDAK punya event operasional sama sekali** di antara
INSTALLED dan akhir siklus. Landmark di sini karena itu GABUNGAN tiga
sumber (`eb_src/landmark_builder.py`):

1. **INSTALL** (age=0) - selalu ada, ekuivalen model statis.
2. **ORGANIC_EVENT** - event operasional nyata di tengah siklus (hanya
   ~20% lifecycle punya ini, rata-rata ~2 event).
3. **ANCHOR** - 90/180/365 hari lalu +365 hari, dibatasi 8 anchor/lifecycle
   (BUKAN grid 30-harian tetap - itu persis pola classification yang ingin
   dihindari sejak awal eksperimen survival).

Split (TRAIN/VALIDATION/TEST) dan cutoff censoring administratif mengikuti
**LIFECYCLE** (installed_on), BUKAN dihitung ulang per-L - keputusan desain
paling penting di modul ini, supaya satu lifecycle TIDAK PERNAH menyebarkan
landmark ke lebih dari satu split (lihat docstring lengkap di
`eb_src/landmark_builder.py`).

Hasil: 23.927 -> 20.116 lifecycle eligible (SAMA dengan model statis, cohort/
censoring tidak diubah) -> **102.728 baris landmark** (TRAIN 92.298/14.980
lifecycle, VALIDATION 5.540/2.316, TEST 4.890/2.820).

## Fitur final (34 kolom - lihat `eb_src/features.py`)

Dasar (sama seperti model statis + umur pemasangan yang di sana di-drop):
riwayat point-in-time, kondisi armada, konteks instalasi (part model/
client/item type), previous-cycle confirmed-failure.

**Dynamic tambahan** (hasil `experiments.py` ablation, `eb_src/dynamic_history.py`):
- **Degradation trend**: rasio interval kegagalan terbaru vs rata-rata
  historis (`failure_interval_trend_ratio`) - di bawah 1 berarti makin
  sering rusak.
- **Cumulative physical usage**: total hari fisik SEMUA siklus sebelumnya
  (`log_cumulative_prior_cycle_days`) + umur fisik total sampai sekarang
  (`log_physical_age_now`) - part yang sudah beberapa kali reinstall TIDAK
  lagi dianggap seperti baru.
- **Jendela corrective 60/90 hari** - melengkapi `prior_corrective_30d`
  yang sudah ada.

**Device/terminal context** (`terminal_type_grouped`, `eb_src/terminal_context.py`):
tipe device (GATE/CVIM/POS/dst.) tempat PART dipasang. **Query kanonikal
BARU** - lihat bagian di bawah.

## Device/terminal: dari schema riset ke query produksi

Ditemukan lewat sweep skema DB (`schema analytics`, view `eda_part_terminal_cycle_link`)
bahwa relasi PART->TERMINAL SUDAH pernah dibangun di fase riset lama, dan
TERBUKTI membantu (+0,007-0,019 tergantung kombinasi) - tapi `config.py`
proyek ini eksplisit melarang production bergantung ke schema `analytics`
("supaya production tidak bergantung sama sekali pada schema analytics
hasil research"), dan `data_reader.py` sendiri mencatat hierarchy TERMINAL
"SENGAJA tidak dibawa" dari riset (README root, diuji dulu untuk model
classification dan "belum terbukti cukup bernilai" di sana - berbeda hasil
dengan event-based di sini, arsitektur model beda bisa dapat sinyal beda
dari fitur yang sama).

**Direproduksi APA ADANYA** sebagai `data_reader.get_terminal_context()`:
definisi VIEW riset dibaca lewat `pg_get_viewdef()` (bukan ditebak), lalu
dibangun ulang dari TABEL MENTAH -
`journal.t_item_request_out` (PART yang diminta keluar gudang, mencatat
`parent_serial_code` device tujuannya) + `master.t_mtr_item` (kategori/tipe
resmi) + `inventory.t_item` (verifikasi device itu benar ada). Diverifikasi
angkanya **PERSIS SAMA** dengan schema `analytics`: 24.008/24.045 valid
link, 10.313 baris "recorded after installation", distribusi terminal_type
identik. Production **TIDAK lagi bergantung ke schema `analytics` sama
sekali**.

Point-in-time: HANYA baris `parent_link_quality_status ==
'VALID_POINT_IN_TIME_RELATION'` (56,9%) yang dipakai sebagai fitur - relasi
yang baru "diketahui" SETELAH instalasi (43%) diberi UNKNOWN, bukan diam-
diam dipakai seolah sudah diketahui sejak awal. Lihat docstring lengkap di
`data_reader.get_terminal_context()`.

**Catatan metodologis**: nilai ablation awal untuk kombinasi ini
(`reports/dynamic_ablation.md`, F_combined_all VAL t0-only=0,8036) memakai
`categorical_support.cumulative_support()` langsung pada frame landmark
untuk menghitung dukungan kategori - bug yang SAMA seperti yang
didokumentasikan di `eb_src/features.py` (menghitung landmark BERKALI-KALI
seolah instalasi baru, bukan sekali per lifecycle). Angka produksi final di
atas (0,7985) memakai `point_in_time_support()` yang BENAR - sedikit lebih
rendah dari ablation awal, tapi metodologinya konsisten dengan seluruh
fitur kategorikal lain. Ablation script (`experiments.py`) TIDAK diperbaiki/
dijalankan ulang (sudah selesai tugasnya mengarahkan keputusan), dicatat di
sini supaya perbedaan angka tidak membingungkan.

## Kenapa ada 3 lapis evaluasi (bukan 2 seperti model statis)

- **Lapis 1 (full landmark)**: C-index dari SEMUA baris landmark. Metrik
  paling dekat dengan cara `predict.py` dipakai di produksi (skor PART pada
  umur berapa pun sekarang) - TAPI baris-baris satu lifecycle SALING
  BERKORELASI (repeated measures), C-index naif di populasi ini BISA bias
  optimis, BUKAN perbandingan apples-to-apples dengan model statis.
- **Lapis 1b (t0-only)**: subset HANYA baris `landmark_source=='INSTALL'` -
  SATU baris per lifecycle, populasi IDENTIK dengan C-index model statis.
  **Ini angka yang sah dibandingkan head-to-head.**
- **Lapis 2**: perbandingan operasional vs classification production, pada
  fitur t0-only (alasan sama - lihat `evaluate.py` docstring).

## Concept drift (jendela tahun TRAIN)

Diuji (`reports/concept_drift.md`): TRAIN 2018-2024/2022-2024 memberi VAL
t0-only ~0,806-0,807, lebih tinggi dari TRAIN 2014-2024 penuh (0,796) -
TAPI polanya TIDAK monoton (2020-2024 turun lagi ke 0,796) dan besarnya
sepadan dengan noise (std bootstrap baseline ~0,01). **TIDAK diadopsi** -
sinyal campuran, butuh validasi multi-seed sebelum jadi perubahan permanen
pada data TRAIN produksi.

## Belum dikerjakan (kandidat lanjutan)

- **Evaluasi "satu landmark acak per lifecycle"**: metrik ketiga yang
  menghindari BAIK inflasi repeated-measures (Lapis 1) MAUPUN pesimisme
  "selalu umur nol" (Lapis 1b) - lebih dekat merepresentasikan pemakaian
  produksi nyata.
- **GBSA pada data landmark** (Tahap 10) - dicoba sekali, terlalu lambat
  (>1 jam CPU pada fitur gabungan) untuk lingkungan ini, dihentikan.
  RSF+Cox tetap yang dipakai.
- Validasi concept drift multi-seed sebelum adopsi.
- Threshold kategori (200/300) dipakai APA ADANYA dari model statis, belum
  di-sweep ulang khusus populasi landmark.

## Rekomendasi saat ini

**Layak jadi challenger kedua** (di samping static) - BUKAN pengganti.
Pada C-index murni masih di bawah static, tapi pada metrik operasional
(PR-AUC, Recall@kapasitas) sudah lebih baik. Arsitektur landmark (censoring
per-split ikut lifecycle, reuse total `attach_history`/`attach_fleet`,
device/terminal via query kanonikal sendiri) valid, tidak ditemukan
leakage, dan TIDAK bergantung ke schema riset mana pun.

## Struktur

```
event_based/
├── README.md
├── build_dataset.py           # DB -> lifecycle -> landmark -> fitur dinamis -> split
├── train.py                     # latih RSF + Cox PH (reuse src/model_fit.py PARENT)
├── evaluate.py                    # 3 lapis (full/t0-only/operasional)
├── predict.py                       # CLI: skor PART pada KONDISI SEKARANG
├── experiments.py                     # ablation A-G (dynamic history + device/terminal)
├── experiments_g_only.py                # re-run cepat 1 konfigurasi
├── experiments_round2.py                  # GBSA + concept drift (lihat catatan lambat)
├── concept_drift.py                         # sweep jendela tahun TRAIN, fitur final
├── eb_src/                        # BUKAN "src" - nama package survival_model/src
│   │                              # sudah dipakai, py tidak bisa punya 2 modul
│   │                              # bernama sama aktif bersamaan (lihat catatan
│   │                              # sys.path di tiap file)
│   ├── landmark_builder.py         # desain landmark, REUSE total logic censoring
│   ├── features.py                   # fitur final + orkestrasi attach_*
│   ├── dynamic_history.py              # degradation trend, cumulative usage, jendela corrective
│   └── terminal_context.py               # device/terminal, point-in-time filtered
├── artifacts/
└── reports/
```

## Cara pakai

```bash
python survival_model/event_based/build_dataset.py
python survival_model/event_based/train.py
python survival_model/event_based/evaluate.py
python survival_model/event_based/predict.py <item_id>
```
