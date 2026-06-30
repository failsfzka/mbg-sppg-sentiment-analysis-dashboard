import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import WARNA_SENTIMEN, WARNA_SARKASME, WARNA_THREAT
from utils.data_loader import load_labeling_data
from components.kpi_card import render_kpi_row

URUTAN_THREAT = ["No Threat (Dukungan Positif)", "Low Threat (Monitoring)", "Medium Threat (Waspada)", "High Threat (Kritis)"]


def render_kpi_model(df_labeling_data):
    '''Menampilkan kartu KPI dominan dari keempat dimensi label model.'''
    modus_sentimen = df_labeling_data["sentiment_prediction"].value_counts().idxmax()
    persen_modus_sentimen = df_labeling_data["sentiment_prediction"].value_counts(normalize=True).max() * 100
    persen_sarkasme = (df_labeling_data["sarcasm_prediction"] == "Sarkasme").mean() * 100
    modus_isu = df_labeling_data["issue_category"].value_counts().idxmax()
    jumlah_high_threat = int((df_labeling_data["threat_level"] == "High Threat (Kritis)").sum())

    daftar_kpi = [
        {"icon": "😊", "label": "Sentimen Dominan", "value": f"{modus_sentimen} ({persen_modus_sentimen:.0f}%)"},
        {"icon": "😏", "label": "Persentase Sarkasme", "value": f"{persen_sarkasme:.1f}%"},
        {"icon": "🗂️", "label": "Isu Dominan", "value": modus_isu},
        {"icon": "🚨", "label": "Komentar Risiko Tinggi", "value": f"{jumlah_high_threat:,}"},
    ]
    render_kpi_row(daftar_kpi, jumlah_kolom=4)


def render_distribusi_sentimen(df_labeling_data):
    '''Menampilkan pie chart distribusi sentimen.'''
    dist = df_labeling_data["sentiment_prediction"].value_counts().reset_index()
    dist.columns = ["sentimen", "jumlah"]
    fig = px.pie(dist, names="sentimen", values="jumlah", color="sentimen",
                 color_discrete_map=WARNA_SENTIMEN, title="Pie Chart — Distribusi Sentimen")
    st.plotly_chart(fig, use_container_width=True)


def render_distribusi_sarkasme(df_labeling_data):
    '''Menampilkan donut chart distribusi sarkasme.'''
    dist = df_labeling_data["sarcasm_prediction"].value_counts().reset_index()
    dist.columns = ["sarkasme", "jumlah"]
    fig = px.pie(dist, names="sarkasme", values="jumlah", hole=0.45, color="sarkasme",
                 color_discrete_map=WARNA_SARKASME, title="Donut Chart — Distribusi Sarkasme")
    st.plotly_chart(fig, use_container_width=True)


def render_distribusi_issue(df_labeling_data):
    '''Menampilkan bar chart distribusi kategori isu.'''
    dist = df_labeling_data["issue_category"].value_counts().reset_index()
    dist.columns = ["isu", "jumlah"]
    fig = px.bar(dist.sort_values("jumlah"), x="jumlah", y="isu", orientation="h",
                 color_discrete_sequence=["#1B5E20"], title="Bar Chart — Distribusi Kategori Isu")
    st.plotly_chart(fig, use_container_width=True)


def render_distribusi_threat(df_labeling_data):
    '''Menampilkan bar chart distribusi tingkat ancaman, terurut sesuai keparahan.'''
    dist = df_labeling_data["threat_level"].value_counts().reindex(URUTAN_THREAT).reset_index()
    dist.columns = ["threat_level", "jumlah"]
    fig = px.bar(dist, x="threat_level", y="jumlah", color="threat_level",
                 color_discrete_map=WARNA_THREAT, title="Bar Chart — Distribusi Tingkat Ancaman")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Komentar")
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap_sentimen_sarkasme(df_labeling_data):
    '''Menampilkan heatmap relasi Sentimen x Sarkasme.'''
    crosstab = pd.crosstab(df_labeling_data["sentiment_prediction"], df_labeling_data["sarcasm_prediction"])
    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values, x=crosstab.columns.tolist(), y=crosstab.index.tolist(),
        colorscale="YlOrRd", text=crosstab.values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Heatmap Relasi Sentimen x Sarkasme", xaxis_title="Sarkasme", yaxis_title="Sentimen")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RENDER HALAMAN MODEL ANALYTICS
# ============================================================
st.markdown("<div class='page-header-title'>🤖 Model Analytics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Ringkasan distribusi hasil klasifikasi: Sentimen, Sarkasme, Isu, dan Tingkat Ancaman</div>",
    unsafe_allow_html=True,
)

df_labeling_halaman = load_labeling_data()

if df_labeling_halaman.empty:
    st.warning("Dataset labeling belum tersedia. Pastikan berkas data/dataset_labeling.csv telah disiapkan.")
else:
    render_kpi_model(df_labeling_halaman)
    st.divider()

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        render_distribusi_sentimen(df_labeling_halaman)
        render_distribusi_issue(df_labeling_halaman)
    with kolom_kanan:
        render_distribusi_sarkasme(df_labeling_halaman)
        render_distribusi_threat(df_labeling_halaman)

    st.divider()
    st.markdown("##### 🔥 Heatmap Relasi Sentimen x Sarkasme")
    render_heatmap_sentimen_sarkasme(df_labeling_halaman)
