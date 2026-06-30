import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import WARNA_SARKASME
from utils.data_loader import load_labeling_data
from components.kpi_card import render_kpi_row


def render_kpi_sarkasme(df_labeling_data):
    '''Menampilkan kartu KPI persentase & jumlah sarkasme.'''
    persentase_sarkasme = (df_labeling_data["sarcasm_prediction"] == "Sarkasme").mean() * 100
    jumlah_sarkasme = int((df_labeling_data["sarcasm_prediction"] == "Sarkasme").sum())
    jumlah_non_sarkasme = int((df_labeling_data["sarcasm_prediction"] == "Non-Sarkasme").sum())

    silang = pd.crosstab(df_labeling_data["sentiment_prediction"], df_labeling_data["sarcasm_prediction"], normalize="index") * 100
    sentimen_sarkasme_tertinggi = silang["Sarkasme"].idxmax() if "Sarkasme" in silang.columns else "-"

    daftar_kpi = [
        {"icon": "😏", "label": "Persentase Sarkasme", "value": f"{persentase_sarkasme:.1f}%"},
        {"icon": "🗯️", "label": "Jumlah Komentar Sarkasme", "value": f"{jumlah_sarkasme:,}"},
        {"icon": "💬", "label": "Jumlah Non-Sarkasme", "value": f"{jumlah_non_sarkasme:,}"},
        {"icon": "🎯", "label": "Sentimen Paling Sarkastik", "value": sentimen_sarkasme_tertinggi},
    ]
    render_kpi_row(daftar_kpi, jumlah_kolom=4)


def render_distribusi_sarkasme(df_labeling_data):
    '''Menampilkan pie chart distribusi sarkasme umum dan bar chart per sentimen.'''
    kolom_kiri, kolom_kanan = st.columns(2)

    with kolom_kiri:
        dist = df_labeling_data["sarcasm_prediction"].value_counts().reset_index()
        dist.columns = ["sarkasme", "jumlah"]
        fig_pie = px.pie(dist, names="sarkasme", values="jumlah", color="sarkasme",
                          color_discrete_map=WARNA_SARKASME, title="Pie Chart — Distribusi Sarkasme")
        st.plotly_chart(fig_pie, use_container_width=True)

    with kolom_kanan:
        silang = pd.crosstab(df_labeling_data["sentiment_prediction"], df_labeling_data["sarcasm_prediction"]).reset_index()
        silang_panjang = silang.melt(id_vars="sentiment_prediction", var_name="sarkasme", value_name="jumlah")
        fig_bar = px.bar(silang_panjang, x="sentiment_prediction", y="jumlah", color="sarkasme",
                          color_discrete_map=WARNA_SARKASME, barmode="group",
                          title="Bar Chart — Distribusi Sarkasme per Sentimen")
        fig_bar.update_layout(xaxis_title="Sentimen", yaxis_title="Jumlah Komentar")
        st.plotly_chart(fig_bar, use_container_width=True)


def render_heatmap_sarkasme_isu(df_labeling_data):
    '''Menampilkan heatmap persentase sarkasme per kategori isu.'''
    silang_persen = pd.crosstab(
        df_labeling_data["issue_category"], df_labeling_data["sarcasm_prediction"], normalize="index"
    ) * 100
    silang_persen = silang_persen.sort_values("Sarkasme", ascending=False) if "Sarkasme" in silang_persen.columns else silang_persen

    fig = go.Figure(data=go.Heatmap(
        z=silang_persen.values, x=silang_persen.columns.tolist(), y=silang_persen.index.tolist(),
        colorscale="Oranges", text=silang_persen.round(1).values, texttemplate="%{text}%",
    ))
    fig.update_layout(title="Heatmap Persentase Sarkasme per Kategori Isu", xaxis_title="Sarkasme", yaxis_title="Kategori Isu")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RENDER HALAMAN SARCASM ANALYTICS
# ============================================================
st.markdown("<div class='page-header-title'>😏 Sarcasm Analytics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Analisis persentase, distribusi, dan relasi sarkasme dengan sentimen & isu kebijakan</div>",
    unsafe_allow_html=True,
)

df_labeling_halaman = load_labeling_data()

if df_labeling_halaman.empty:
    st.warning("Dataset labeling belum tersedia. Pastikan berkas data/dataset_labeling.csv telah disiapkan.")
else:
    render_kpi_sarkasme(df_labeling_halaman)
    st.divider()
    render_distribusi_sarkasme(df_labeling_halaman)
    st.divider()
    st.markdown("##### 🔥 Heatmap Sarkasme x Kategori Isu")
    render_heatmap_sarkasme_isu(df_labeling_halaman)
