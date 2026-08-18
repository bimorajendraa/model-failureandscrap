"""Penilaian lengkap satu PART."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import ui

ui.page_setup("Detail PART")
ui.sidebar_status()

st.title("Detail PART")
st.caption("Masukkan Item ID. Seluruh fitur ML dibangun otomatis dari riwayat PART.")

# Diisi otomatis kalau datang dari tombol "Lihat detail" di tabel manapun
# (lihat ui.priority_table). Dikeluarkan dari session_state sekali pakai
# supaya kembali ke halaman ini secara manual tidak terus mengunci PART lama.
default_item_id = st.session_state.pop("detail_item_id", "")

item_id = st.text_input(
    "Item ID", value=default_item_id, placeholder="mis. 011201100101164"
).strip()
if not item_id:
    st.stop()

try:
    data = api_client.assessment(item_id)
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()

if data.get("status") == "NOT_FOUND":
    st.error(data.get("message", f"PART '{item_id}' tidak ditemukan."))
    st.stop()

if data.get("status") == "NOT_SCORABLE":
    st.warning(f"**Tidak bisa dinilai.** {data.get('reason', '')}")
    st.caption(
        "Ini bukan kegagalan sistem: PART yang sedang tidak terpasang memang "
        "tidak punya risiko kerusakan yang perlu diperkirakan. Data yang tidak "
        "ada sengaja TIDAK diganti angka karangan agar prediksi tetap keluar."
    )
    st.stop()

failure = data["failure"]
scrap = data["scrap"]
recommendation = data["recommendation"]

st.subheader(f"PART {data['item_id']}")
info = st.columns(4)
info[0].markdown(f"**Jenis PART**\n\n{(scrap or {}).get('item_type') or '-'}")
info[1].markdown(f"**Kelompok risiko kerusakan**\n\n{ui.risk_badge(failure['risk_level'])}")
info[2].markdown(
    f"**Kelompok risiko scrap**\n\n{ui.risk_badge((scrap or {}).get('scrap_risk_level'))}"
)
info[3].markdown(f"**Data sampai**\n\n{data['as_of']}")

st.divider()
st.subheader("Risiko kerusakan")
ui.horizon_metrics(failure)
ui.probability_caption()

st.subheader("Risiko scrap")
if scrap:
    left, right = st.columns([1, 2])
    left.metric("Tidak bisa diperbaiki jika rusak", api_client.percent(scrap["scrap_probability"]))
    if data.get("death_probability_30d") is not None:
        left.metric(
            "Peluang PART mati dalam 30 hari",
            api_client.percent(data["death_probability_30d"]),
            help="Peluang rusak dalam 30 hari x peluang tidak bisa diperbaiki.",
        )
    right.info(
        f"Angka ini **bersyarat**: {scrap['scrap_risk_basis']}. "
        "Bukan peluang PART ini rusak."
    )
    if not scrap.get("item_type_known_to_model", True):
        right.warning(
            "Jenis PART ini belum dikenal model scrap, jadi dinilai bersama "
            "kelompok jenis yang jarang - angkanya perlu dibaca lebih hati-hati."
        )
else:
    st.info("Riwayat PART ini belum cukup untuk dinilai model scrap.")

st.divider()
st.subheader("Rekomendasi")
st.markdown(
    f"### {recommendation['action'].replace('_', ' ').title()}\n"
    f"Prioritas: {ui.risk_badge(recommendation['priority'])}"
)
st.write(recommendation["message"])
if data.get("replacement_candidate"):
    st.warning(
        "PART ini masuk **kandidat penggantian** - risiko rusak dan risiko "
        "scrap sama-sama tinggi. Bukan vonis bahwa PART akan dibuang."
    )

explanation = data.get("explanation")
if explanation:
    st.divider()
    st.subheader("Faktor risiko")
    st.caption(explanation["disclaimer"])
    icons = {"RISK_FACTOR": "🔺", "MITIGATING": "🟢", "CONTEXT": "•"}
    for factor in explanation["factors"]:
        st.markdown(f"{icons.get(factor['direction'], '•')} {factor['label']}")
    for note in explanation.get("notes", []):
        st.caption(note)
    for caveat in explanation["caveats"]:
        st.warning(caveat)

    try:
        history = api_client.history(data["item_id"])
    except api_client.ApiError as error:
        history = None
        st.caption(f"Riwayat detail tidak bisa dimuat: {error}")

    if history:
        with st.expander("Tanggal kerusakan"):
            if history["failures"]:
                st.dataframe(
                    pd.DataFrame(history["failures"]).rename(
                        columns={"date": "Tanggal", "location": "Lokasi", "status": "Status"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Seluruh kerusakan yang tercatat, bukan hanya 365 hari terakhir."
                )
            else:
                st.caption("Belum pernah tercatat rusak.")

        with st.expander("Riwayat lokasi"):
            if history["locations"]:
                st.dataframe(
                    pd.DataFrame(history["locations"]).rename(columns={
                        "location": "Lokasi",
                        "first_seen": "Pertama tercatat",
                        "last_seen": "Terakhir tercatat",
                        "events": "Jumlah catatan",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("Diurutkan dari lokasi yang paling terakhir aktif.")
            else:
                st.caption("Belum ada lokasi yang tercatat.")

st.divider()
versions = data["model_version"]
st.caption(
    f"Model kerusakan **{versions['failure']}** · model scrap "
    f"**{versions['scrap'] or '-'}**"
)
