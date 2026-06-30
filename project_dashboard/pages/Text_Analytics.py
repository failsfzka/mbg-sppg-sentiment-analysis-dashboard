import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from utils.data_loader import load_eda_data, pilih_kolom_teks


def kolom_tersedia(dataframe_target, nama_kolom):
    '''Mengembalikan True apabila nama_kolom tersedia pada dataframe_target.'''
    return nama_kolom in dataframe_target.columns


def hitung_top_ngram(daftar_teks, ukuran_ngram, jumlah_teratas=15):
    '''Menghitung n-gram (bigram/trigram) paling sering muncul dari sekumpulan teks.'''
    vectorizer = CountVectorizer(ngram_range=(ukuran_ngram, ukuran_ngram), max_features=2000)
    matriks_frekuensi = vectorizer.fit_transform(daftar_teks)
    total_frekuensi = matriks_frekuensi.sum(axis=0).A1
    nama_ngram = vectorizer.get_feature_names_out()
    pasangan_ngram = sorted(zip(nama_ngram, total_frekuensi), key=lambda x: -x[1])
    return pasangan_ngram[:jumlah_teratas]


def render_wordcloud(teks_gabungan):
    '''Menghasilkan dan menampilkan gambar WordCloud dari teks gabungan.'''
    generator_wordcloud = WordCloud(
        width=900, height=450, background_color="white",
        colormap="Greens", max_words=120, collocations=False,
    ).generate(teks_gabungan)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.imshow(generator_wordcloud, interpolation="bilinear")
    axis.axis("off")
    st.pyplot(figure)


def render_top_words(teks_series):
    '''Menampilkan grafik batang 15 kata paling sering muncul.'''
    semua_kata = [kata for teks in teks_series for kata in str(teks).split() if len(kata) > 2]
    frekuensi_kata = Counter(semua_kata)
    df_top_words = pd.DataFrame(frekuensi_kata.most_common(15), columns=["kata", "frekuensi"])
    fig = px.bar(
        df_top_words.sort_values("frekuensi"), x="frekuensi", y="kata", orientation="h",
        color_discrete_sequence=["#1B5E20"], title="Top 15 Kata Paling Sering Muncul",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_ngram(teks_series, ukuran_ngram, judul, warna):
    '''Menampilkan grafik batang top-15 n-gram (bigram/trigram).'''
    hasil_ngram = hitung_top_ngram(teks_series, ukuran_ngram=ukuran_ngram)
    df_ngram = pd.DataFrame(hasil_ngram, columns=["frasa", "frekuensi"])
    fig = px.bar(
        df_ngram.sort_values("frekuensi"), x="frekuensi", y="frasa", orientation="h",
        color_discrete_sequence=[warna], title=judul,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_histogram_panjang(teks_series):
    '''Menampilkan histogram panjang komentar (jumlah kata).'''
    panjang_komentar = teks_series.str.split().apply(len)
    fig = px.histogram(
        panjang_komentar, nbins=40, color_discrete_sequence=["#1B5E20"],
        labels={"value": "Jumlah Kata per Komentar"}, title="Histogram Panjang Komentar",
    )
    fig.update_layout(showlegend=False, yaxis_title="Frekuensi")
    st.plotly_chart(fig, use_container_width=True)
    kolom_a, kolom_b = st.columns(2)
    kolom_a.metric("Rata-rata Panjang Komentar", f"{panjang_komentar.mean():.1f} kata")
    kolom_b.metric("Median Panjang Komentar", f"{panjang_komentar.median():.1f} kata")


def render_timeline(df_eda_data):
    '''Menampilkan timeline jumlah komentar per hari dan per jam.'''
    if not kolom_tersedia(df_eda_data, "published_at"):
        st.info("Kolom 'published_at' tidak ditemukan, timeline tidak dapat ditampilkan.")
        return

    tanggal = pd.to_datetime(df_eda_data["published_at"], errors="coerce")
    komentar_per_hari = tanggal.dt.date.value_counts().sort_index().reset_index()
    komentar_per_hari.columns = ["tanggal", "jumlah"]
    komentar_per_jam = tanggal.dt.hour.value_counts().sort_index().reset_index()
    komentar_per_jam.columns = ["jam", "jumlah"]

    st.markdown("##### Jumlah Komentar per Hari")
    fig_harian = px.line(komentar_per_hari, x="tanggal", y="jumlah", markers=True, color_discrete_sequence=["#1B5E20"])
    st.plotly_chart(fig_harian, use_container_width=True)

    st.markdown("##### Jumlah Komentar per Jam")
    fig_jam = px.bar(komentar_per_jam, x="jam", y="jumlah", color_discrete_sequence=["#FF8F00"])
    fig_jam.update_layout(xaxis_title="Jam (0-23)", yaxis_title="Jumlah Komentar")
    st.plotly_chart(fig_jam, use_container_width=True)


# ============================================================
# RENDER HALAMAN TEXT ANALYTICS
# ============================================================
st.markdown("<div class='page-header-title'>📝 Text Analytics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Analisis frekuensi kata, frasa, panjang komentar, dan timeline Dataset EDA</div>",
    unsafe_allow_html=True,
)

df_eda_halaman = load_eda_data()

if df_eda_halaman.empty:
    st.warning("Dataset EDA belum tersedia. Pastikan berkas data/dataset_eda.csv telah disiapkan.")
else:
    kolom_teks_analisis = pilih_kolom_teks(df_eda_halaman, ["stopword_removal", "normalisasi", "text_clean", "comment"])

    if kolom_teks_analisis is None:
        st.error("Tidak ditemukan kolom teks yang dapat dianalisis pada Dataset EDA.")
    else:
        teks_series_halaman = df_eda_halaman[kolom_teks_analisis].dropna().astype(str)
        teks_gabungan_halaman = " ".join(teks_series_halaman.tolist())

        tab_kata, tab_panjang, tab_timeline = st.tabs(["☁️ Kata & Frasa", "📏 Panjang Komentar", "🕒 Timeline"])

        with tab_kata:
            render_wordcloud(teks_gabungan_halaman)
            kolom_kiri, kolom_kanan = st.columns(2)
            with kolom_kiri:
                render_top_words(teks_series_halaman)
            with kolom_kanan:
                render_top_ngram(teks_series_halaman, ukuran_ngram=2, judul="Top 15 Bigram", warna="#3498DB")
            render_top_ngram(teks_series_halaman, ukuran_ngram=3, judul="Top 15 Trigram", warna="#E67E22")

        with tab_panjang:
            render_histogram_panjang(teks_series_halaman)

        with tab_timeline:
            render_timeline(df_eda_halaman)
