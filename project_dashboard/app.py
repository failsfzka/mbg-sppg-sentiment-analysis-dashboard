import streamlit as st
from config import APP_TITLE, APP_ICON, APP_LAYOUT, PATH_CUSTOM_CSS, DAFTAR_HALAMAN


def muat_css_kustom(path_css):
    '''Membaca berkas CSS kustom dan menyuntikkannya ke halaman Streamlit.'''
    try:
        with open(path_css, "r", encoding="utf-8") as file_css:
            st.markdown(f"<style>{file_css.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Berkas styles/custom.css tidak ditemukan, dashboard menggunakan tema default.")


# Konfigurasi halaman utama (harus dipanggil pertama kali sebelum elemen Streamlit lainnya)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

muat_css_kustom(PATH_CUSTOM_CSS)

# Mendaftarkan seluruh halaman dashboard menggunakan API navigasi Streamlit
daftar_objek_halaman = [
    st.Page(item["file"], title=item["title"], icon=item["icon"])
    for item in DAFTAR_HALAMAN
]

navigasi = st.navigation(daftar_objek_halaman)
navigasi.run()
