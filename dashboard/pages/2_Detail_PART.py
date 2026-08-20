"""Penilaian lengkap satu PART."""

from __future__ import annotations

import streamlit as st

import api_client
import ui

ui.page_setup("Detail PART")
ui.sidebar_status()

st.title("Detail PART")
st.caption("Masukkan Item ID. Seluruh fitur ML dibangun otomatis dari riwayat PART.")

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
        "tidak punya risiko kerusakan yang perlu diperkirakan."
    )
    st.stop()

failure = data["failure"]
scrap = data["scrap"]
recommendation = data["recommendation"]

with st.container(border=True):
    st.subheader(f"PART {data['item_id']}")
    info = st.columns(4)
    info[0].markdown(f"**Jenis PART**\n\n{(scrap or {}).get('item_type') or '-'}")
    info[1].markdown(f"**Risiko kerusakan**\n\n{ui.risk_badge(failure['risk_level'])}")
    info[2].markdown(
        f"**Risiko rusak total**\n\n{ui.risk_badge((scrap or {}).get('scrap_risk_level'))}"
    )
    info[3].markdown(f"**Data sampai**\n\n{data['as_of']}")

st.subheader("Risiko kerusakan")
with st.container(border=True):
    ui.horizon_metrics(failure)
    ui.probability_caption()

st.subheader("Risiko rusak total")
st.caption(
    "Kalau PART ini rusak, seberapa besar kemungkinan sudah tidak bisa "
    "diperbaiki lagi (harus diganti baru)."
)
with st.container(border=True):
    if scrap:
        left, right = st.columns([1, 2])
        left.metric("Peluang rusak total (jika rusak)", api_client.percent(scrap["scrap_probability"]))
        if data.get("death_probability_30d") is not None:
            left.metric(
                "Peluang harus diganti dalam 30 hari",
                api_client.percent(data["death_probability_30d"]),
                help="Peluang rusak dalam 30 hari dikali peluang rusak total kalau sampai rusak.",
            )
        right.info(
            f"Angka ini **bersyarat** - dihitung {scrap['scrap_risk_basis']}. "
            "Bukan peluang PART ini akan rusak."
        )
        if not scrap.get("item_type_known_to_model", True):
            right.warning(
                "Jenis PART ini masih jarang ditemui model, jadi dinilai bersama "
                "kelompok jenis yang jarang - angkanya perlu dibaca lebih hati-hati."
            )
    else:
        st.info("Riwayat PART ini belum cukup untuk dinilai risiko rusak totalnya.")

st.subheader("Rekomendasi")
with st.container(border=True):
    st.markdown(
        f"### {ui.action_label(recommendation['action'])}\n"
        f"Prioritas: {ui.priority_badge(recommendation['priority'])}"
    )
    st.write(recommendation["message"])
    if data.get("replacement_candidate"):
        st.warning(
            "PART ini masuk **kandidat penggantian** - risiko kerusakan dan "
            "risiko rusak total sama-sama tinggi. Bukan vonis bahwa PART akan dibuang."
        )

explanation = data.get("explanation")
if explanation:
    st.subheader("Faktor risiko")
    with st.container(border=True):
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
        with st.expander("📅 Tanggal kerusakan"):
            ui.labeled_table(
                history["failures"],
                columns=["date", "location", "status"],
                empty_message="Belum pernah tercatat rusak.",
            )
            if history["failures"]:
                st.caption("Seluruh kerusakan yang tercatat, bukan hanya 365 hari terakhir.")

        with st.expander("📍 Riwayat lokasi"):
            ui.labeled_table(
                history["locations"],
                columns=["location", "first_seen", "last_seen", "events"],
                empty_message="Belum ada lokasi yang tercatat.",
            )
            if history["locations"]:
                st.caption("Diurutkan dari lokasi yang paling terakhir aktif.")

st.divider()
versions = data["model_version"]
st.caption(
    f"Model kerusakan **{versions['failure']}** · model rusak total "
    f"**{versions['scrap'] or '-'}**"
)
