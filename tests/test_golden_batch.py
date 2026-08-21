"""Golden batch oracle - bukti bahwa langkah restrukturisasi survival_model
(lihat plan restrukturisasi) yang diberi label MOVE (git mv + rewrite import)
BENAR-BENAR tidak mengubah satu angka pun.

`.cache/golden/phase0_baseline.parquet` dibuat SEKALI di awal restrukturisasi
lewat `python scripts/golden_batch.py generate --out .cache/golden/phase0_baseline.parquet`
(lihat docstring skrip itu untuk kenapa generate-lalu-bandingkan, bukan satu
snapshot beku permanen - database ini live).

Test ini SENGAJA tidak otomatis regenerate baseline-nya sendiri: kalau baseline
hilang/basi, itu harus keputusan sadar (jalankan skrip generate lagi), bukan
diam-diam ditimpa oleh test run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import needs_database, needs_models

BASELINE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "golden" / "phase0_baseline.parquet"


@needs_database
@needs_models
def test_golden_baseline_exists():
    """Kalau ini gagal: jalankan
    `python scripts/golden_batch.py generate --out .cache/golden/phase0_baseline.parquet`
    dulu sebelum melanjutkan langkah MOVE mana pun - tanpa baseline, tidak ada
    yang bisa dibuktikan."""
    assert BASELINE_PATH.exists(), (
        f"Golden baseline belum ada di {BASELINE_PATH}. Jalankan "
        "scripts/golden_batch.py generate sebelum melakukan langkah MOVE."
    )


@needs_database
@needs_models
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="golden baseline belum ada - lihat test_golden_baseline_exists")
def test_current_batch_matches_golden_baseline(batch):
    """Skor batch LIVE (sekarang) harus identik dengan baseline - kolom
    `rank` dikecualikan (urutan bisa goyah kalau ada tier_score yang persis
    sama, bukan tanda kerusakan).

    Kalau test ini gagal SETELAH sebuah langkah MOVE: langkah itu bukan pure
    move, ada perubahan perilaku yang menyelinap - JANGAN lanjut ke langkah
    berikutnya sebelum menemukan penyebabnya.

    Kalau test ini gagal TANPA ada langkah MOVE yang baru dijalankan: populasi
    PART aktif di database sudah berubah sejak baseline dibuat (part baru
    dipasang/rusak) - itu bukan bug, tapi baseline perlu di-generate ulang.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import golden_batch

    # Bukan pytest tmp_path (gagal di beberapa mesin Windows karena izin
    # direktori temp) - pakai scratch di bawah .cache/ yang sudah gitignored,
    # ditimpa tiap run.
    live_path = BASELINE_PATH.parent / "_live_comparison_scratch.parquet"
    golden_batch.generate(live_path)

    assert golden_batch.compare(BASELINE_PATH, live_path), (
        "Batch live BERBEDA dari golden baseline - lihat detail perbedaan di output di atas. "
        "Kalau ini terjadi setelah langkah MOVE, langkah itu bukan pure move."
    )
