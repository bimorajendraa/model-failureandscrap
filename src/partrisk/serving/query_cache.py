"""Menyatukan pembacaan data_reader yang berulang di dalam SATU request.

`predict()`, `predict_scrap()`, dan `predict_survival` (advisory, mode
aditif) masing-masing berdiri sendiri (sengaja, supaya ketiga model tetap
independen), tapi ketiganya membaca get_dataset_max_event_on/get_cycles/
get_events dengan argumen yang sama, dan predict_survival tambah butuh
get_failure_episodes/get_terminal_context. Tanpa penyatuan ini, satu
endpoint assessment memakai belasan koneksi database; dengan ini, jauh lebih
sedikit.

Cache hanya hidup selama `request_scope()` dan hanya untuk thread itu - di
luar scope, data_reader dipanggil apa adanya, jadi predict.py/train.py dari
terminal dan batch scoring tidak terpengaruh. Kesetaraan hasilnya dijaga
tests/test_freshness.py.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager

from partrisk.data import reader as data_reader

# Fungsi baca yang argumennya menentukan hasil sepenuhnya, jadi aman diulang.
_CACHEABLE = (
    "get_dataset_max_event_on", "get_events", "get_cycles",
    "get_failure_episodes", "get_terminal_context",
)

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
