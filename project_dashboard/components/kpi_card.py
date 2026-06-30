import streamlit as st


def render_kpi_card(icon, label, value, help_text=None):
    '''
    Menampilkan satu kartu KPI bergaya kustom menggunakan HTML/CSS.

    Parameter:
        icon (str)       : Emoji ikon yang ditampilkan di atas nilai.
        label (str)       : Label/nama metrik (contoh: "Total Komentar").
        value (str)       : Nilai metrik yang ditampilkan besar (contoh: "69.079").
        help_text (str)   : Teks bantuan opsional yang muncul sebagai tooltip ikon info.
    '''
    html_kartu = f'''
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    '''
    st.markdown(html_kartu, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def render_kpi_row(daftar_kpi, jumlah_kolom=4):
    '''
    Menampilkan beberapa kartu KPI sejajar dalam beberapa kolom.

    Parameter:
        daftar_kpi (list[dict]) : Setiap dict berisi key 'icon', 'label', 'value', dan opsional 'help_text'.
        jumlah_kolom (int)       : Jumlah kolom tampilan, default 4.
    '''
    kolom = st.columns(jumlah_kolom)
    for index, item_kpi in enumerate(daftar_kpi):
        with kolom[index % jumlah_kolom]:
            render_kpi_card(
                icon=item_kpi.get("icon", "📌"),
                label=item_kpi.get("label", ""),
                value=item_kpi.get("value", "-"),
                help_text=item_kpi.get("help_text"),
            )


def render_badge(teks, warna_hex):
    '''Mengembalikan string HTML badge berwarna untuk menandai kategori (sentimen/sarkasme/ancaman).'''
    return f'<span class="badge-pill" style="background-color:{warna_hex};">{teks}</span>'
