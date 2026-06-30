import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from config import WARNA_SENTIMEN
from utils.data_loader import load_labeling_data

PETA_COLORMAP_WORDCLOUD = {"Positif": "Greens", "Netral": "YlOrBr", "Negatif": "Reds"}


def render_distribusi_sentimen(df_labeling_data):
    '''Menampilkan pie chart dan bar chart distribusi sentimen secara berdampingan.'''
    dist = df_labeling_data["sentiment_prediction"].value_counts().reset_index()
    dist.columns = ["sentimen", "jumlah"]

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        fig_pie = px.pie(dist, names="sentimen", values="jumlah", color="sentimen",
                          color_discrete_map=WARNA_SENTIMEN, title="Pie Chart — Distribusi Sentimen")
        st.plotly_chart(fig_pie, use_container_width=True)
    with kolom_kanan:
        fig_bar = px.bar(dist, x="sentimen", y="jumlah", color="sentimen",
                          color_discrete_map=WARNA_SENTIMEN, title="Bar Chart — Distribusi Sentimen")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


def render_wordcloud_per_sentimen(df_labeling_data):
    '''Menampilkan WordCloud terpisah untuk masing-masing kelas sentimen.'''
    kolom = st.columns(3)
    for index, kelas_sentimen in enumerate(["Positif", "Netral", "Negatif"]):
        teks_kelas = " ".join(
            df_labeling_data.loc[df_labeling_data["sentiment_prediction"] == kelas_sentimen, "text_clean"]
            .dropna().astype(str)
        )
        with kolom[index]:
            st.markdown(f"**WordCloud — {kelas_sentimen}**")
            if teks_kelas.strip():
                wc_kelas = WordCloud(
                    width=500, height=350, background_color="white",
                    colormap=PETA_COLORMAP_WORDCLOUD[kelas_sentimen], max_words=80, collocations=False,
                ).generate(teks_kelas)
                figure, axis = plt.subplots(figsize=(5, 3.5))
                axis.imshow(wc_kelas, interpolation="bilinear")
                axis.axis("off")
                st.pyplot(figure)
            else:
                st.info(f"Tidak ada data untuk kelas {kelas_sentimen}.")


def render_top_komentar(df_labeling_data):
    '''Menampilkan tabel 5 komentar Positif dan Negatif terpanjang sebagai komentar representatif.'''
    df_labeling_data = df_labeling_data.copy()
    df_labeling_data["panjang_teks"] = df_labeling_data["text_clean"].astype(str).str.len()

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        st.markdown("##### 🟢 Top 5 Komentar Positif")
        top_positif = (
            df_labeling_data[df_labeling_data["sentiment_prediction"] == "Positif"]
            .sort_values("panjang_teks", ascending=False)
            .head(5)[["text_clean"]]
        )
        st.dataframe(top_positif, use_container_width=True, hide_index=True)

    with kolom_kanan:
        st.markdown("##### 🔴 Top 5 Komentar Negatif")
        top_negatif = (
            df_labeling_data[df_labeling_data["sentiment_prediction"] == "Negatif"]
            .sort_values("panjang_teks", ascending=False)
            .head(5)[["text_clean"]]
        )
        st.dataframe(top_negatif, use_container_width=True, hide_index=True)


# ============================================================
# RENDER HALAMAN SENTIMENT ANALYTICS
# ============================================================
st.markdown("<div class='page-header-title'>😊 Sentiment Analytics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Analisis mendalam distribusi, kata kunci, dan komentar representatif per kelas sentimen</div>",
    unsafe_allow_html=True,
)

df_labeling_halaman = load_labeling_data()

if df_labeling_halaman.empty:
    st.warning("Dataset labeling belum tersedia. Pastikan berkas data/dataset_labeling.csv telah disiapkan.")
else:
    render_distribusi_sentimen(df_labeling_halaman)
    st.divider()
    st.markdown("##### ☁️ WordCloud per Kelas Sentimen")
    render_wordcloud_per_sentimen(df_labeling_halaman)
    st.divider()
    st.markdown("##### 💬 Komentar Representatif")
    st.caption("Definisi 'Top': 5 komentar dengan jumlah karakter terbanyak pada masing-masing kelas.")
    render_top_komentar(df_labeling_halaman)
