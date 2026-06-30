import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import WARNA_THREAT, WARNA_SENTIMEN, DAFTAR_KATEGORI_ISU
from utils.data_loader import load_labeling_data
from components.kpi_card import render_kpi_row

URUTAN_THREAT = [
    "No Threat (Dukungan Positif)",
    "Low Threat (Monitoring)",
    "Medium Threat (Waspada)",
    "High Threat (Kritis)",
]


def render_kpi_policy(df_labeling_data):
    '''Menampilkan 4 KPI ringkasan risiko kebijakan.'''
    total = len(df_labeling_data)
    high = int((df_labeling_data["threat_level"] == "High Threat (Kritis)").sum())
    medium = int((df_labeling_data["threat_level"] == "Medium Threat (Waspada)").sum())
    isu_paling_berisiko = (
        df_labeling_data[df_labeling_data["threat_level"] == "High Threat (Kritis)"]["issue_category"]
        .value_counts().idxmax() if high > 0 else "-"
    )
    render_kpi_row([
        {"icon": "🚨", "label": "Ancaman Kritis (High)", "value": f"{high:,}"},
        {"icon": "⚠️", "label": "Ancaman Waspada (Medium)", "value": f"{medium:,}"},
        {"icon": "🗂️", "label": "Isu Paling Berisiko", "value": isu_paling_berisiko},
        {"icon": "💬", "label": "Total Komentar", "value": f"{total:,}"},
    ], jumlah_kolom=4)


def render_treemap_isu(df_labeling_data):
    '''Menampilkan Treemap distribusi komentar per isu kebijakan.'''
    df_treemap = df_labeling_data.groupby(
        ["issue_category", "threat_level"]
    ).size().reset_index(name="jumlah")

    fig = px.treemap(
        df_treemap, path=["issue_category", "threat_level"],
        values="jumlah", color="threat_level",
        color_discrete_map=WARNA_THREAT,
        title="Treemap: Distribusi Komentar per Isu dan Tingkat Ancaman",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_stacked_bar_isu_threat(df_labeling_data):
    '''Menampilkan Stacked Bar Chart Isu × Tingkat Ancaman.'''
    ct = pd.crosstab(df_labeling_data["issue_category"], df_labeling_data["threat_level"])
    kolom_terurut = [k for k in URUTAN_THREAT if k in ct.columns]
    ct = ct[kolom_terurut].reset_index()
    ct_long = ct.melt(id_vars="issue_category", var_name="threat_level", value_name="jumlah")
    ct_long["threat_level"] = pd.Categorical(ct_long["threat_level"], categories=kolom_terurut, ordered=True)
    ct_long = ct_long.sort_values("threat_level")

    fig = px.bar(
        ct_long, x="issue_category", y="jumlah",
        color="threat_level", color_discrete_map=WARNA_THREAT,
        title="Stacked Bar: Komposisi Ancaman per Kategori Isu",
        category_orders={"threat_level": URUTAN_THREAT},
    )
    fig.update_layout(xaxis_title="Kategori Isu", yaxis_title="Jumlah Komentar",
                      legend_title="Tingkat Ancaman")
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap_isu_sentimen(df_labeling_data):
    '''Heatmap interaktif Isu × Sentimen.'''
    ct = pd.crosstab(df_labeling_data["issue_category"], df_labeling_data["sentiment_prediction"])
    fig = go.Figure(data=go.Heatmap(
        z=ct.values, x=list(ct.columns), y=list(ct.index),
        colorscale="RdYlGn",
        text=[[f"{v:,}" for v in row] for row in ct.values],
        texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        title="Heatmap: Isu Kebijakan x Sentimen",
        xaxis_title="Prediksi Sentimen", yaxis_title="Kategori Isu", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap_isu_threat(df_labeling_data):
    '''Heatmap interaktif Isu × Tingkat Ancaman.'''
    ct = pd.crosstab(df_labeling_data["issue_category"], df_labeling_data["threat_level"])
    kolom_terurut = [k for k in URUTAN_THREAT if k in ct.columns]
    ct = ct[kolom_terurut]
    fig = go.Figure(data=go.Heatmap(
        z=ct.values, x=list(ct.columns), y=list(ct.index),
        colorscale="Reds",
        text=[[f"{v:,}" for v in row] for row in ct.values],
        texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        title="Heatmap: Isu Kebijakan x Tingkat Ancaman",
        xaxis_title="Tingkat Ancaman", yaxis_title="Kategori Isu", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_threat_per_sentimen(df_labeling_data):
    '''Bar Chart jumlah komentar per tingkat ancaman, dipilah berdasarkan sentimen.'''
    ct = pd.crosstab(df_labeling_data["threat_level"], df_labeling_data["sentiment_prediction"])
    ct = ct.reindex(URUTAN_THREAT).dropna(how="all").reset_index()
    ct_long = ct.melt(id_vars="threat_level", var_name="sentimen", value_name="jumlah")
    fig = px.bar(
        ct_long, x="threat_level", y="jumlah",
        color="sentimen", color_discrete_map=WARNA_SENTIMEN, barmode="group",
        title="Ancaman per Tingkat, Dipilah Berdasarkan Sentimen",
        category_orders={"threat_level": URUTAN_THREAT},
    )
    fig.update_layout(xaxis_title="Tingkat Ancaman", yaxis_title="Jumlah Komentar")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RENDER HALAMAN POLICY RISK ANALYTICS
# ============================================================
st.markdown("<div class='page-header-title'>⚠️ Policy Risk Analytics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Pemetaan risiko kebijakan: isu strategis, tingkat ancaman, dan pola sentimen</div>",
    unsafe_allow_html=True,
)
df_lb = load_labeling_data()
if df_lb.empty:
    st.warning("Dataset labeling belum tersedia.")
else:
    render_kpi_policy(df_lb)
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Treemap", "📊 Stacked Bar", "🔥 Heatmap Isu x Sentimen", "❗ Heatmap Isu x Ancaman"]
    )
    with tab1:
        render_treemap_isu(df_lb)
        render_threat_per_sentimen(df_lb)
    with tab2:
        render_stacked_bar_isu_threat(df_lb)
    with tab3:
        render_heatmap_isu_sentimen(df_lb)
    with tab4:
        render_heatmap_isu_threat(df_lb)
