# Threshold kategori khusus survival

Threshold KHUSUS survival (bukan `config.MIN_PART_MODEL_SUPPORT=300` classification, yang dikalibrasi untuk skala 251.568 baris TRAIN classification, bukan ~15rb lifecycle TRAIN survival). Dipilih dari VAL C-index (RSF ringan, 50 pohon) - TEST TIDAK dipakai memilih, hanya dilaporkan pada tahap ablation/final.

| Kolom | Threshold | Kategori asli | Digabung LOW_SUPPORT | Unseen VAL | Unseen TEST | VAL C-index (RSF ringan) |
|---|---|---|---|---|---|---|
| item_model_code_clean | 20 | 46 | 3 | 353/2316 | 369/2820 | 0.8040 |
| item_model_code_clean | 50 | 46 | 9 | 458/2316 | 639/2820 | 0.8081 |
| item_model_code_clean | 100 | 46 | 13 | 593/2316 | 693/2820 | 0.8114 |
| item_model_code_clean | 200 | 46 | 22 | 919/2316 | 852/2820 | 0.8116 **<-dipilih** |
| item_model_code_clean | 300 | 46 | 28 | 1060/2316 | 1016/2820 | 0.8084 |
| item_type_at_install | 20 | 18 | 0 | 0/2316 | 0/2820 | 0.8102 |
| item_type_at_install | 50 | 18 | 1 | 6/2316 | 3/2820 | 0.8095 |
| item_type_at_install | 100 | 18 | 1 | 6/2316 | 3/2820 | 0.8093 |
| item_type_at_install | 200 | 18 | 4 | 66/2316 | 71/2820 | 0.8096 |
| item_type_at_install | 300 | 18 | 6 | 100/2316 | 100/2820 | 0.8147 **<-dipilih** |
| place_at_install | 20 | 137 | 24 | 852/2316 | 239/2820 | 0.8056 |
| place_at_install | 50 | 137 | 50 | 1013/2316 | 330/2820 | 0.8099 **<-dipilih** |
| place_at_install | 100 | 137 | 78 | 1322/2316 | 874/2820 | 0.8055 |
| place_at_install | 200 | 137 | 121 | 1842/2316 | 1604/2820 | 0.8044 |
| place_at_install | 300 | 137 | 130 | 2107/2316 | 2520/2820 | 0.8081 |

Threshold terpilih: `{'item_model_code_clean': 200, 'item_type_at_install': 300, 'place_at_install': 50}`