"""Menghapus pembacaan database yang terduplikasi di dalam SATU request.

MASALAH YANG DIPECAHKAN DI SINI

Menilai satu PART memanggil `predict()` dan `predict_scrap()`, dan keduanya -
masing-masing berdiri sendiri, sebagaimana mestinya - membaca hal yang sama:

    predict()        get_dataset_max_event_on(), get_cycles(item), get_events(item)
    predict_scrap()  get_dataset_max_event_on(), get_events(item), get_cycles(item)

Argumennya identik, hasilnya identik, tetapi setiap panggilan membuka koneksi
baru. Terukur: 9 koneksi dan 9 detik untuk satu endpoint assessment.

Menyatukannya di dalam ML core berarti membongkar pemisahan kedua model, dan
itu justru yang harus dijaga. Jadi penyatuannya dilakukan di sini: selama satu
request, pembacaan dengan argumen yang sama dijawab dari hasil pertama.

BATASNYA JELAS, DAN ITU YANG MEMBUATNYA AMAN

- Cache hanya hidup di dalam `request_scope()`, dan hanya untuk thread itu.
  Di luar scope, fungsi aslinya dipanggil apa adanya - `train.py`, `predict.py`
  dari terminal, dan batch scoring tidak terpengaruh sama sekali.
- Isinya dibuang begitu scope selesai, jadi tidak mungkin ada data yang
  bertahan antar-request.
- Dalam satu request, batas waktu data memang HARUS sama untuk kedua model.
  Membacanya dua kali justru berisiko menghasilkan dua titik observasi berbeda
  kalau database bertambah di tengah request.

Kesetaraan hasilnya dijaga tests/test_query_cache.py, yang membandingkan
prediksi dengan dan tanpa cache untuk PART yang sama.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager

import data_reader

# Fungsi baca yang argumennya menentukan hasil sepenuhnya, jadi aman diulang.
_CACHEABLE = ("get_dataset_max_event_on", "get_events", "get_cycles")

_local = threading.local()
_installed = False


def _scope() -> dict | None:
    return getattr(_local, "scope", None)


def _wrap(name: str, original):
    @functools.wraps(original)
    def reader(*args, **kwargs):
        scope = _scope()
        if scope is None:
            return original(*args, **kwargs)

        key = (name, args, tuple(sorted(kwargs.items())))
        if key not in scope:
            scope[key] = original(*args, **kwargs)
        return scope[key]

    reader.__wrapped_by_query_cache__ = True
    return reader


def install() -> None:
    """Pasang pembungkus pada data_reader. Aman dipanggil berulang."""
    global _installed
    if _installed:
        return
    for name in _CACHEABLE:
        original = getattr(data_reader, name)
        if getattr(original, "__wrapped_by_query_cache__", False):
            continue
        setattr(data_reader, name, _wrap(name, original))
    _installed = True


@contextmanager
def request_scope():
    """Satukan pembacaan berulang selama blok ini berjalan.

    Scope bersarang tidak membuat cache baru - yang terluar yang memiliki dan
    membuangnya, sehingga hasilnya tetap konsisten di seluruh blok.
    """
    install()
    if _scope() is not None:
        yield
        return

    _local.scope = {}
    try:
        yield
    finally:
        _local.scope = None


def reads_in_scope() -> int:
    """Berapa pembacaan berbeda yang tersimpan. Dipakai test."""
    scope = _scope()
    return 0 if scope is None else len(scope)
