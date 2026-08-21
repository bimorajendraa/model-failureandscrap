"""Golden batch oracle - alat verifikasi untuk restrukturisasi survival_model
(lihat plan restrukturisasi, Fase 0.2).

MASALAH yang diselesaikan: ~135 test yang ada di-skip kalau database/model
tidak tersedia, dan bahkan yang jalan tidak membandingkan OUTPUT PERSIS
sebelum/sesudah sebuah langkah MOVE (git mv + rewrite import, seharusnya nol
perubahan logika). Tanpa oracle ini, tidak ada cara membuktikan bahwa
memindahkan file tidak diam-diam mengubah angka.

KENAPA generate-lalu-bandingkan, BUKAN satu snapshot beku: database ini live
(PART baru dipasang, kerusakan baru tercatat setiap hari) - membandingkan
batch hari ini dengan batch minggu lalu akan menunjukkan "perbedaan" yang
sebenarnya cuma data asli berubah, bukan bug. Jadi pola pakainya:

    python scripts/golden_batch.py generate --out before.parquet   # SEBELUM langkah MOVE
    ...lakukan git mv + rewrite import...
    python scripts/golden_batch.py generate --out after.parquet    # SESUDAH, secepat mungkin
    python scripts/golden_batch.py compare before.parquet after.parquet

Rentang waktu antara generate before/after harus sesingkat mungkin (idealnya
di bawah beberapa menit) supaya drift data asli tidak tercampur dengan bukti
yang sedang dicari.

    python scripts/golden_batch.py generate --out .cache/golden/phase0_baseline.parquet
    python scripts/golden_batch.py compare a.parquet b.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Kolom yang SENGAJA dikecualikan dari perbandingan nilai - bukan bug kalau
# beda, memang seharusnya beda antar dua kali generate.
_VOLATILE_COLUMNS = {"rank"}  # urutan bisa goyah kalau ada dua tier_score persis sama (tie-break tidak stabil)


def _load_batch():
    from inference import batch_predictor

    return batch_predictor.score_active_parts(force_refresh=True)


def generate(out_path: Path) -> None:
    print(f"[1/2] Menjalankan batch_predictor.score_active_parts(force_refresh=True)...")
    t0 = time.time()
    batch = _load_batch()
    print(f"      selesai dalam {time.time()-t0:.1f} detik - {len(batch.frame):,} PART aktif")

    print(f"[2/2] Menyimpan ke {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = batch.frame.copy()
    frame.attrs.clear()
    snapshot = batch.snapshot.reset_index().rename(columns={"index": "item_id"})

    # Satu file parquet, dua tabel dibedakan kolom "_table" - lebih sederhana
    # daripada dua file yang bisa saling terpisah.
    frame.insert(0, "_table", "frame")
    snapshot.insert(0, "_table", "snapshot")
    combined = pd.concat([frame, snapshot], axis=0, ignore_index=True, sort=False)
    combined.to_parquet(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.txt")
    meta_path.write_text(
        f"generated_at={pd.Timestamp.now(tz='UTC').isoformat()}\n"
        f"data_end={batch.data_end}\n"
        f"model_version={batch.model_version}\n"
        f"rows_frame={len(batch.frame)}\n"
        f"rows_snapshot={len(batch.snapshot)}\n",
        encoding="utf-8",
    )
    print(f"      OK - {meta_path}")


def _split(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = combined.loc[combined["_table"] == "frame"].drop(columns=["_table"]).dropna(axis=1, how="all")
    snapshot = combined.loc[combined["_table"] == "snapshot"].drop(columns=["_table"]).dropna(axis=1, how="all")
    return frame.reset_index(drop=True), snapshot.reset_index(drop=True)


def compare(path_a: Path, path_b: Path, *, rtol: float = 1e-9) -> bool:
    """True kalau IDENTIK (di luar _VOLATILE_COLUMNS). Mencetak diagnosa
    kalau tidak."""
    frame_a, snap_a = _split(pd.read_parquet(path_a))
    frame_b, snap_b = _split(pd.read_parquet(path_b))

    ok = True
    for name, a, b, key in (("frame", frame_a, frame_b, "item_id"), ("snapshot", snap_a, snap_b, "item_id")):
        print(f"\n--- {name}: {path_a.name} ({len(a):,} baris) vs {path_b.name} ({len(b):,} baris) ---")
        cols_a, cols_b = set(a.columns), set(b.columns)
        if cols_a != cols_b:
            print(f"  KOLOM BEDA: hanya di A={cols_a-cols_b}  hanya di B={cols_b-cols_a}")
            ok = False
            continue

        a_sorted = a.sort_values(key).reset_index(drop=True)
        b_sorted = b.sort_values(key).reset_index(drop=True)
        if list(a_sorted[key]) != list(b_sorted[key]):
            only_a = set(a_sorted[key]) - set(b_sorted[key])
            only_b = set(b_sorted[key]) - set(a_sorted[key])
            print(f"  POPULASI {key} BEDA: hanya di A={len(only_a)}  hanya di B={len(only_b)}")
            if only_a:
                print(f"    contoh hanya-A: {list(only_a)[:5]}")
            if only_b:
                print(f"    contoh hanya-B: {list(only_b)[:5]}")
            ok = False
            common = sorted(set(a_sorted[key]) & set(b_sorted[key]))
            a_sorted = a_sorted.set_index(key).loc[common].reset_index()
            b_sorted = b_sorted.set_index(key).loc[common].reset_index()

        for col in sorted(cols_a - _VOLATILE_COLUMNS):
            sa, sb = a_sorted[col], b_sorted[col]
            if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
                diff_mask = ~np.isclose(
                    sa.to_numpy(dtype=float), sb.to_numpy(dtype=float), rtol=rtol, equal_nan=True
                )
            else:
                diff_mask = (sa.astype(str) != sb.astype(str)).to_numpy()
            n_diff = int(diff_mask.sum())
            if n_diff:
                ok = False
                idx = np.flatnonzero(diff_mask)[:5]
                sample = [
                    (a_sorted[key].iloc[i], sa.iloc[i], sb.iloc[i]) for i in idx
                ]
                print(f"  KOLOM '{col}': {n_diff}/{len(a_sorted):,} baris beda. Contoh (id, A, B): {sample}")

    print(f"\n{'=== IDENTIK ===' if ok else '=== ADA PERBEDAAN - lihat di atas ==='}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--out", type=Path, required=True)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("path_a", type=Path)
    p_cmp.add_argument("path_b", type=Path)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.out)
        return 0
    if args.command == "compare":
        return 0 if compare(args.path_a, args.path_b) else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
