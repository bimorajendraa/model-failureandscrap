"""Kesalahan yang dikenal lapisan serving.

Dipisah dari service supaya route dan dashboard bisa membedakan "PART tidak
ada" dari "PART ada tapi tidak bisa diskor" tanpa membaca teks pesan.
"""

from __future__ import annotations


class PartNotFound(LookupError):
    """PART tidak ada di database sama sekali."""

    def __init__(self, item_id: str, message: str | None = None) -> None:
        self.item_id = item_id
        self.message = message or f"PART '{item_id}' tidak ditemukan di database."
        super().__init__(self.message)


class PartNotScorable(RuntimeError):
    """PART ada, tetapi kondisinya membuat model tidak bisa memberi angka.

    Bukan kesalahan sistem: PART yang sedang tidak terpasang memang tidak
    punya risiko kerusakan yang perlu diperkirakan. Alasannya dibawa apa
    adanya dari ML core, bukan dikarang ulang di sini.
    """

    def __init__(self, item_id: str, reason: str) -> None:
        self.item_id = item_id
        self.reason = reason
        super().__init__(reason)


class ModelUnavailable(RuntimeError):
    """Model production belum ada atau gagal dimuat."""


class DataSourceUnavailable(RuntimeError):
    """Database tidak bisa dibaca."""
