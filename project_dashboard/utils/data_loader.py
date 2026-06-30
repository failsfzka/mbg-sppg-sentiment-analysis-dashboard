import os
import pandas as pd
import streamlit as st

# Direktori dasar project (induk dari folder utils/), dihitung otomatis agar path tetap valid
# terlepas dari direktori kerja (current working directory) saat Streamlit dijalankan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PATH_EDA = os.path.join(DATA_DIR, "dataset_eda.csv")
PATH_LABELING = os.path.join(DATA_DIR, "dataset_labeling.csv")
PATH_INTEGRATED = os.path.join(DATA_DIR, "dataset_integrated.csv")


def pilih_kolom_teks(dataframe_target, daftar_kandidat):
    '''Mengembalikan nama kolom pertama dari daftar kandidat yang tersedia pada dataframe.'''
    for nama_kolom in daftar_kandidat:
        if nama_kolom in dataframe_target.columns:
            return nama_kolom
    return None


@st.cache_data(show_spinner="Memuat Dataset EDA...")
def load_eda_data():
    '''Memuat Dataset EDA (hasil scraping & preprocessing) dari folder data/.'''
    if not os.path.exists(PATH_EDA):
        st.error(f"Berkas dataset EDA tidak ditemukan di: {PATH_EDA}")
        return pd.DataFrame()
    return pd.read_csv(PATH_EDA)


@st.cache_data(show_spinner="Memuat Dataset Final Labeling...")
def load_labeling_data():
    '''Memuat Dataset Final Labeling (hasil fine-tuning IndoBERT & deteksi sarkasme) dari folder data/.'''
    if not os.path.exists(PATH_LABELING):
        st.error(f"Berkas dataset labeling tidak ditemukan di: {PATH_LABELING}")
        return pd.DataFrame()
    return pd.read_csv(PATH_LABELING)


@st.cache_data(show_spinner="Memuat Integrated Dataset...")
def load_integrated_data():
    '''Memuat Integrated Dataset (gabungan Dataset EDA & Dataset Labeling) dari folder data/.'''
    if not os.path.exists(PATH_INTEGRATED):
        st.error(f"Berkas dataset terintegrasi tidak ditemukan di: {PATH_INTEGRATED}")
        return pd.DataFrame()
    return pd.read_csv(PATH_INTEGRATED)
