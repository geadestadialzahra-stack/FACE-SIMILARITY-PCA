import streamlit as st
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import os
import requests
from urllib.parse import urlparse

# ======================== KONFIGURASI HALAMAN ========================
st.set_page_config(
    page_title="ANGEL APP",
    page_icon="🌸",
    layout="wide"
)

# ======================== CSS GLOBAL ========================
st.markdown("""
    <style>
        /* ----- BACKGROUND & WARNA DASAR ----- */
        .stApp, .main, .block-container, section.main, div[data-testid="stSidebar"] {
            background-color: #FFF0F5 !important;
        }
        body, p, div, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stCaption {
            color: #6A1B4D !important;
        }
        header {
            background: linear-gradient(135deg, #880E4F, #AD1457) !important;
            border-bottom: 2px solid #F8BBD0 !important;
        }
        header * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        .css-1d391kg, .css-12w0qpk, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FCE4EC, #FFF0F5) !important;
            border-right: 2px solid #F8BBD0 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #AD1457 !important;
            font-weight: bold !important;
        }

        /* ----- SIDEBAR DEKORASI ----- */
        .sidebar-header {
            text-align: center;
            padding: 5px 0 5px 0;
            border-bottom: 2px solid #F8BBD0;
            margin-bottom: 10px;
        }
        .sidebar-header .logo {
            font-size: 40px;
            display: block;
            margin-bottom: 2px;
        }
        .sidebar-header .title {
            font-size: 22px;
            font-weight: bold;
            color: #AD1457;
            letter-spacing: 2px;
        }
        .sidebar-header .subtitle {
            font-size: 13px;
            color: #880E4F;
            font-style: italic;
            margin-top: 2px;
        }
        .sidebar-footer {
            text-align: center;
            font-size: 12px;
            color: #AD1457;
            border-top: 1px solid #F8BBD0;
            padding-top: 8px;
            margin-top: 12px;
        }

        /* ----- TOMBOL UMUM ----- */
        .stButton button {
            background: linear-gradient(135deg, #EC407A, #D81B60) !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 50px !important;
            padding: 10px 30px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3) !important;
            transition: 0.3s !important;
        }
        .stButton button:hover {
            transform: scale(1.03) translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(233, 30, 99, 0.4) !important;
        }

        /* ----- FILE UPLOADER ----- */
        div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5) !important;
            border: 2px dashed #EC407A !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
        div[data-testid="stFileUploader"] > div {
            background: rgba(255, 255, 255, 0.6) !important;
            border-radius: 8px !important;
            padding: 20px !important;
        }
        div[data-testid="stFileUploader"] * {
            color: #6A1B4D !important;
            background: transparent !important;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #EC407A, #D81B60) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 5px 20px !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            transform: scale(1.05) !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #D81B60 !important;
        }

        /* ----- SLIDER ----- */
        .stSlider > div {
            background: rgba(255, 255, 255, 0.4) !important;
            border-radius: 10px !important;
        }

        /* ----- SIDEBAR NAVIGASI (tombol bulat) ----- */
        .stSidebar .stButton button {
            width: 48px !important;
            height: 48px !important;
            min-width: 48px !important;
            min-height: 48px !important;
            max-width: 48px !important;
            max-height: 48px !important;
            border-radius: 50% !important;
            border: none !important;
            background: transparent !important;
            font-size: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 auto !important;
            box-shadow: none !important;
            transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            line-height: 1 !important;
        }
        .stSidebar .stButton button:hover {
            transform: scale(1.06) !important;
            background: rgba(236, 64, 122, 0.06) !important;
            box-shadow: 0 0 12px rgba(236, 64, 122, 0.08) !important;
        }
        .sidebar-caption {
            text-align: center;
            color: #AD1457;
            font-weight: bold;
            font-size: 13px;
            padding-top: 3px;
            margin-bottom: 8px;
        }

        /* ----- PROFIL TIM DI SIDEBAR (KOTAK PER ANGGOTA) ----- */
        .sidebar-profile {
            margin-top: 8px;
            padding: 5px 5px;
        }
        .sidebar-profile h4 {
            color: #AD1457;
            text-align: center;
            margin-bottom: 10px;
            font-size: 1rem;
        }
        .sidebar-profile .profile-item {
            display: flex;
            align-items: center;
            margin-bottom: 14px !important;
            padding: 12px 14px;
            border-radius: 14px;
            background: #ffffff !important;
            border: 2px solid #EC407A !important;
            box-shadow: 0 4px 12px rgba(173,20,87,0.15);
            transition: 0.2s;
        }
        .sidebar-profile .profile-item:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 18px rgba(173,20,87,0.25);
        }
        .sidebar-profile .profile-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #EC407A, #D81B60);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            margin-right: 14px;
            flex-shrink: 0;
            overflow: hidden;
            border: 2px solid white;
            box-shadow: 0 2px 8px rgba(173,20,87,0.15);
        }
        .sidebar-profile .profile-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .sidebar-profile .profile-info .name {
            font-weight: bold;
            font-size: 0.95rem;
            color: #6A1B4D;
        }
        .sidebar-profile .profile-info .detail {
            font-size: 0.75rem;
            color: #880E4F;
            margin-top: 2px;
        }
        .sidebar-university {
            text-align: center;
            padding: 8px 0;
            color: #AD1457;
            font-weight: bold;
            font-size: 0.9rem;
            border-top: 1px solid #F8BBD0;
            margin-top: 8px;
        }

        /* ----- GAYA KONTEN UTAMA ----- */
        .content-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(173,20,87,0.08);
            border: 1px solid #F8BBD0;
        }
        .content-card h2 {
            color: #AD1457;
            margin-top: 0;
        }
        .image-card {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(173,20,87,0.1);
            border: 1px solid #F8BBD0;
            margin: 0.5rem 0;
        }
        .image-card img {
            border-radius: 12px;
            border: 2px solid #F8BBD0;
            width: 100%;
        }
        .download-btn {
            background: linear-gradient(135deg, #EC407A, #D81B60);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.6rem 2rem;
            font-weight: bold;
            transition: 0.3s;
            box-shadow: 0 4px 15px rgba(233,30,99,0.3);
            cursor: pointer;
        }
        .download-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(233,30,99,0.4);
        }
        .info-box {
            background: rgba(255,255,255,0.7);
            border-left: 5px solid #EC407A;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 2px 10px rgba(233,30,99,0.08);
        }
        .info-box b {
            color: #AD1457;
        }

        /* ----- KOTAK KETERANGAN TAMBAHAN (di bawah setiap halaman) ----- */
        .footer-note {
            background: linear-gradient(135deg, #FFF9C4, #FFE082);
            border-radius: 16px;
            padding: 1.2rem 2rem;
            margin-top: 2rem;
            border: 1px solid #FFB300;
            text-align: center;
            color: #BF360C;
        }
        .footer-note p {
            margin: 0;
            font-size: 1rem;
        }

        /* ----- GAYA KHUSUS HOME (bling-bling) ----- */
        .home-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FFF9C4, #FFE082);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 2px solid #FFB300;
            box-shadow: 0 0 30px rgba(255, 193, 7, 0.2);
            animation: glowPulse 2s ease-in-out infinite alternate;
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 20px rgba(255, 193, 7, 0.1); }
            100% { box-shadow: 0 0 50px rgba(255, 193, 7, 0.4); }
        }
        .home-header h1 {
            font-size: 2.8rem;
            color: #E65100;
            margin: 0;
            font-weight: 900;
            text-shadow: 0 0 20px rgba(255, 193, 7, 0.3);
        }
        .bling-shower {
            font-size: 2rem;
            letter-spacing: 6px;
            color: #FFB300;
            animation: sparkle 1.5s ease-in-out infinite alternate;
        }
        @keyframes sparkle {
            0% { opacity: 0.4; transform: scale(0.95); }
            100% { opacity: 1; transform: scale(1.1); }
        }

        /* ----- GAYA KHUSUS GRAYSCALE (bunga) ----- */
        .grayscale-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #F8BBD0;
        }
        .grayscale-header h1 {
            font-size: 2.5rem;
            color: #AD1457;
            margin: 0;
            font-weight: 800;
        }
        .flower-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #EC407A;
            animation: twinkle 2s infinite alternate;
        }
        @keyframes twinkle {
            0% { opacity: 0.6; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.05); }
        }

        /* ----- GAYA KHUSUS KOMPRESI (awan) ----- */
        .kompresi-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #90CAF9;
        }
        .kompresi-header h1 {
            font-size: 2.5rem;
            color: #0D47A1;
            margin: 0;
            font-weight: 800;
        }
        .cloud-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #42A5F5;
            animation: floatCloud 3s ease-in-out infinite alternate;
        }
        @keyframes floatCloud {
            0% { transform: translateY(0); }
            100% { transform: translateY(-8px); }
        }

        /* ----- GAYA KHUSUS DETEKSI (love) ----- */
        .deteksi-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #F8BBD0;
        }
        .deteksi-header h1 {
            font-size: 2.5rem;
            color: #AD1457;
            margin: 0;
            font-weight: 800;
        }
        .love-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #EC407A;
            animation: pulseLove 1.5s ease-in-out infinite alternate;
        }
        @keyframes pulseLove {
            0% { transform: scale(1); opacity: 0.7; }
            100% { transform: scale(1.1); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ======================== SESSION STATE ========================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "home_visited" not in st.session_state:
    st.session_state.home_visited = False
if "grayscale_visited" not in st.session_state:
    st.session_state.grayscale_visited = False
if "kompresi_visited" not in st.session_state:
    st.session_state.kompresi_visited = False
if "deteksi_visited" not in st.session_state:
    st.session_state.deteksi_visited = False

# ======================== FUNGSI BANTU UNTUK FOTO ========================
def get_image_base64(path_or_url):
    try:
        parsed = urlparse(path_or_url)
        if parsed.scheme in ('http', 'https'):
            response = requests.get(path_or_url, timeout=5)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode()
        else:
            if os.path.exists(path_or_url):
                with open(path_or_url, "rb") as f:
                    return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

# ======================== SIDEBAR NAVIGASI & PROFIL ========================
# --- HEADER SIDEBAR ---
st.sidebar.markdown("""
<div class="sidebar-header">
    <span class="logo">🌸</span>
    <span class="title">ANGEL</span>
    <div class="subtitle">✨ Edit & Kreasikan Gambarmu ✨</div>
</div>
""", unsafe_allow_html=True)

# --- PESAN DI BAWAH HEADER ---
st.sidebar.markdown("""
<div style="text-align: center; font-size: 14px; color: #880E4F; padding: 0 5px 8px 5px; font-style: italic;">
    Lupakan dia yang membuatmu terluka,<br>semoga web ini bisa membuatmu bahagia. <br> <br> Silahkan pilih menu yang diinginkan !
</div>
""", unsafe_allow_html=True)

# --- MENU NAVIGASI ---
menus = [
    ("🏠", "🏠 Home", "Home"),
    ("🌫️", "🌫️ Grayscale", "Grayscale"),
    ("🗜️", "🗜️ Kompresi", "Kompresi"),
    ("🔍", "🔍 Deteksi", "Deteksi")
]

cols = st.sidebar.columns(4)
for col, (emoji, page_name, label) in zip(cols, menus):
    with col:
        is_active = (st.session_state.page == page_name)
        if is_active:
            st.markdown(f"""
                <style>
                    .stSidebar .stButton button[data-testid="baseButton-secondary"]:has(> div:contains("{emoji}")) {{
                        background: #F8BBD0 !important;
                        transform: translateY(2px) scale(1.03) !important;
                        box-shadow: 0 4px 14px rgba(236,64,122,0.25) !important;
                        border: none !important;
                    }}
                </style>
            """, unsafe_allow_html=True)
        if st.button(emoji, key=f"nav_{emoji}", use_container_width=True):
            st.session_state.page = page_name
            # Reset efek
            if page_name == "🏠 Home":
                st.session_state.home_visited = False
            elif page_name == "🌫️ Grayscale":
                st.session_state.grayscale_visited = False
            elif page_name == "🗜️ Kompresi":
                st.session_state.kompresi_visited = False
            elif page_name == "🔍 Deteksi":
                st.session_state.deteksi_visited = False
            st.rerun()

# --- CAPTION DI BAWAH TOMBOL ---
st.sidebar.markdown("---")
if st.session_state.page == "🏠 Home":
    st.sidebar.markdown('<p class="sidebar-caption">🏠 Home</p>', unsafe_allow_html=True)
elif st.session_state.page == "🌫️ Grayscale":
    st.sidebar.markdown('<p class="sidebar-caption">🌫️ Grayscale</p>', unsafe_allow_html=True)
elif st.session_state.page == "🗜️ Kompresi":
    st.sidebar.markdown('<p class="sidebar-caption">🗜️ Kompresi</p>', unsafe_allow_html=True)
elif st.session_state.page == "🔍 Deteksi":
    st.sidebar.markdown('<p class="sidebar-caption">🔍 Deteksi Kemiripan</p>', unsafe_allow_html=True)

# ======================== PROFIL TIM DI SIDEBAR (dengan kotak tegas) ========================
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
st.sidebar.markdown("### 👥 Pengembangan Aplikasi")
st.sidebar.markdown("**Teknik Informatika**")

# DATA ANGGOTA
anggota = [
    {
    
        "nama": "Gea Destadia Al-Zahra",
        "ig": "@gea_destadia_10",
        "telp": "0831-5068-7481",
        "foto": "assets/gea.jpg"
    },
    {
        
        "nama": "Luna Amilia",
        "ig": "@luunaaamiiii",
        "telp": "0895-3780-96802",
        "foto": "assets/luna.jpg"
    },
    {
    
        "nama": "Nadia Azizah",
        "ig": "@ndyyzh",
        "telp": "0858-4631-3309",
        "foto": "assets/nadia.jpg"
    },
    {

        "nama": "Dalilah Arifah Ariandi DJR",
        "ig": "@adellianav",
        "telp": "0813-1211-6787",
        "foto": "assets/dalilah.jpg"
    },
]

for member in anggota:
    foto_b64 = get_image_base64(member.get("foto", ""))
    if foto_b64:
        avatar_html = f'<img src="data:image/jpeg;base64,{foto_b64}" />'
    else:
        avatar_html = member["inisial"]
    
    st.sidebar.markdown(f"""
    <div class="profile-item">
        <div class="profile-avatar">{avatar_html}</div>
        <div class="profile-info">
            <div class="name">• {member['nama']} •</div>
            <div class="detail">📸 {member['ig']}</div>
            <div class="detail">📞 {member['telp']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-university">🎓 Universitas Negeri Semarang</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER SIDEBAR ---
st.sidebar.markdown("""
<div class="sidebar-footer">
    🌸 Made with Love by Team ANGEL 🌸
</div>
""", unsafe_allow_html=True)


# ======================== HALAMAN UTAMA ========================
page = st.session_state.page

if page == "🏠 Home":
    # ==================== HOME ====================
    if not st.session_state.home_visited:
        st.balloons()
        st.session_state.home_visited = True

    st.markdown("""
    <div class="home-header">
        <div class="bling-shower">✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨</div>
        <h1>🌸 Selamat Datang di ANGEL 🌸</h1>
        <p style="font-size:1.3rem; color:#BF360C; font-weight:500;">
            Tempat terbaik untuk mengolah gambar Anda dengan sentuhan kecantikan.
        </p>
        <div class="bling-shower">✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FFF9C4, #FFE082); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #FFB300; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#E65100;">
            🌟 <b>ANGEL</b> hadir untuk membantu Anda mengubah gambar menjadi lebih artistik, 
            ringkas, dan bermakna. Jelajahi fitur-fitur kami dan temukan keajaiban visual.
        </p>
        <p style="color:#BF360C; font-style:italic;">
            "Setiap gambar memiliki cerita – biarkan kami membantu Anda menceritakannya."
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <h2>🌸 Fitur Unggulan</h2>
        <p>
            <b>🌫️ Grayscale</b> – Ubah gambar menjadi hitam-putih untuk efek artistik dan penghematan ukuran.<br>
            <b>🗜️ Kompresi PCA</b> – Reduksi dimensi gambar menggunakan Principal Component Analysis, menjaga kualitas visual dengan ukuran lebih kecil.<br>
            <b>🔍 Deteksi Kemiripan</b> – Bandingkan dua gambar dan dapatkan skor kemiripan secara otomatis.
        </p>
        <p>
            <b>Manfaat:</b><br>
            ✅ Menghemat ruang penyimpanan.<br>
            ✅ Mempermudah analisis visual.<br>
            ✅ Hasil cepat dan akurat.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Teknologi untuk kreativitas tanpa batas."
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- KETERANGAN TAMBAHAN DI BAWAH HOME ---
    st.markdown("""
    <div class="footer-note">
        <p>📌 <b>Keterangan:</b> Halaman ini adalah pintu masuk utama. 
        Gunakan menu di sidebar untuk mengakses fitur pengolahan gambar. 
        Selamat berkarya! 🌸</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🌫️ Grayscale":
    # ==================== GRAYSCALE ====================
    if not st.session_state.grayscale_visited:
        st.balloons()
        st.session_state.grayscale_visited = True

    st.markdown("""
    <div class="grayscale-header">
        <div class="flower-shower">🌸 🌺 🌷 🌹 🌻 🌼 🌸 🌺 🌷 🌹 🌻 🌼</div>
        <h1>🌫️ Konversi ke Grayscale</h1>
        <p>Ubah warna menjadi cerita hitam-putih yang abadi.</p>
        <div class="flower-shower">🌸 🌺 🌷 🌹 🌻 🌼 🌸 🌺 🌷 🌹 🌻 🌼</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            🌟 <b>Grayscale</b> adalah seni mengubah spektrum warna menjadi gradasi abu-abu yang elegan. 
            Setiap piksel bercerita tentang kontras, tekstur, dan emosi – tanpa gangguan warna.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Terkadang, hitam-putih justru lebih hidup."
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Unggah gambar (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img1, col_img2 = st.columns(2, gap="medium")

        with col_img1:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
            st.markdown(f"*Ukuran: {image.width} x {image.height} px*")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_img2:
            if st.button("🔄 Konversi ke Grayscale", use_container_width=True):
                gray_image = image.convert("L")
                gray_rgb = gray_image.convert("RGB")

                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.markdown("### ⚫ Hasil Grayscale")
                st.image(gray_rgb, use_container_width=True)
                st.markdown(f"*Ukuran: {gray_rgb.width} x {gray_rgb.height} px*")
                st.markdown('</div>', unsafe_allow_html=True)

                buf = io.BytesIO()
                gray_rgb.save(buf, format="PNG")
                byte_im = buf.getvalue()
                b64 = base64.b64encode(byte_im).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="grayscale.png" style="text-decoration:none;">'
                href += '<button class="download-btn">⬇️ Download Hasil</button></a>'
                st.markdown(href, unsafe_allow_html=True)

                st.success("🌸 Semoga membantu, terima kasih banyak telah menggunakan jasa layanan kami, salam cinta ❤️")
                st.balloons()

                st.markdown("""
                <div class="info-box">
                    <b>💡 Manfaat Grayscale:</b><br>
                    • Mengurangi kompleksitas warna, fokus pada bentuk dan tekstur.<br>
                    • Menghemat ruang penyimpanan (ukuran file lebih kecil).<br>
                    • Memberikan nuansa artistik dan klasik pada foto.
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <p style="font-size:1.2rem; color:#6A1B4D;">👆 Unggah gambar untuk mulai mengubahnya menjadi hitam-putih</p>
            <p style="color:#AD1457; opacity:0.7;">Atau lihat contoh di bawah ini:</p>
        </div>
        """, unsafe_allow_html=True)

        example_img = Image.new('RGB', (400, 300), color='#FCE4EC')
        draw = ImageDraw.Draw(example_img)
        draw.rectangle([50, 50, 150, 150], fill='#EC407A')
        draw.rectangle([200, 50, 300, 150], fill='#42A5F5')
        draw.rectangle([50, 180, 150, 280], fill='#66BB6A')
        draw.rectangle([200, 180, 300, 280], fill='#FFA726')
        st.image(example_img, caption="Contoh gambar (unggah gambar Anda sendiri untuk hasil nyata)", use_container_width=True)

    # --- KETERANGAN TAMBAHAN DI BAWAH GRAYSCALE ---
    st.markdown("""
    <div class="footer-note">
        <p>📌 <b>Keterangan:</b> Fitur ini mengubah gambar berwarna menjadi hitam-putih (grayscale). 
        Hasilnya dapat diunduh dalam format PNG. Cocok untuk efek klasik dan penghematan ukuran file.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🗜️ Kompresi":
    # ==================== KOMPRESI PCA ====================
    if not st.session_state.kompresi_visited:
        st.balloons()
        st.session_state.kompresi_visited = True

    st.markdown("""
    <div class="kompresi-header">
        <div class="cloud-shower">☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️</div>
        <h1>🗜️ Kompresi Gambar dengan PCA</h1>
        <p>Kecilkan ukuran, pertahankan esensi.</p>
        <div class="cloud-shower">☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #90CAF9; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#0D47A1;">
            ☁️ <b>Principal Component Analysis (PCA)</b> adalah teknik cerdas yang mereduksi dimensi gambar 
            tanpa menghilangkan jati dirinya. Seperti awan yang menampung jutaan tetes air, 
            PCA merangkum informasi penting dalam bentuk yang lebih ringkas.
        </p>
        <p style="color:#1565C0; font-style:italic;">
            "Kecilkan ukuran, pertahankan esensi – itulah keajaiban kompresi."
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Unggah gambar untuk dikompresi",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        h, w, c = img_array.shape

        n_components = st.slider(
            "Jumlah komponen PCA (semakin kecil, semakin besar kompresi)",
            min_value=10,
            max_value=min(h, w, 200),
            value=min(h, w, 100),
            step=10
        )

        if st.button("🚀 Kompresi dengan PCA", use_container_width=True):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            flat_img = img_array.reshape(-1, c)
            reduced = pca.fit_transform(flat_img)
            reconstructed = pca.inverse_transform(reduced)
            compressed_img = reconstructed.reshape(h, w, c).astype(np.uint8)
            compressed_pil = Image.fromarray(compressed_img)

            col_ori, col_comp = st.columns(2)
            with col_ori:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Gambar Asli")
                st.image(image, use_container_width=True)
                st.markdown(f"*Ukuran: {w} x {h} px*")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_comp:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.markdown(f"### 🗜️ Hasil Kompresi (n={n_components})")
                st.image(compressed_pil, use_container_width=True)
                st.markdown(f"*Ukuran: {w} x {h} px*")
                st.markdown('</div>', unsafe_allow_html=True)

            buf = io.BytesIO()
            compressed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            b64 = base64.b64encode(byte_im).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="compressed_pca.png" style="text-decoration:none;">'
            href += '<button class="download-btn">⬇️ Download Hasil Kompresi</button></a>'
            st.markdown(href, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <b>💡 Manfaat Kompresi PCA:</b><br>
                • Mengurangi ukuran file secara signifikan (hingga 70% lebih kecil).<br>
                • Mempercepat transfer dan penyimpanan data.<br>
                • Tetap mempertahankan fitur utama gambar – hanya yang tidak penting yang dibuang.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("👆 Unggah gambar untuk memulai kompresi.")

    # --- KETERANGAN TAMBAHAN DI BAWAH KOMPRESI ---
    st.markdown("""
    <div class="footer-note">
        <p>📌 <b>Keterangan:</b> Kompresi PCA mengurangi dimensi gambar dengan mempertahankan informasi paling penting. 
        Geser slider untuk mengatur tingkat kompresi. Hasil dapat diunduh sebagai PNG.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔍 Deteksi":
    # ==================== DETEKSI KEMIRIPAN ====================
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Gambar</h1>
        <p>Temukan koneksi visual di antara dua gambar.</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Kemiripan</b> adalah jembatan antara dua gambar. Dengan <b>SSIM</b> dan <b>MSE</b>, 
            kita mengukur seberapa dekat mereka dalam struktur, kontras, dan kecerahan. 
            Setiap angka bercerita tentang persamaan yang mungkin tak terlihat oleh mata.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Dua gambar mungkin berbeda, tetapi jiwa visualnya bisa sama."
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Gambar pertama", type=["jpg", "jpeg", "png"], key="img1")
    with col_upload2:
        img2 = st.file_uploader("📤 Gambar kedua", type=["jpg", "jpeg", "png"], key="img2")

    if img1 is not None and img2 is not None:
        image1 = Image.open(img1).convert("RGB")
        image2 = Image.open(img2).convert("RGB")
        size = (300, 300)
        im1 = image1.resize(size)
        im2 = image2.resize(size)

        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(im1, caption="Gambar 1", use_container_width=True)
        with col_show2:
            st.image(im2, caption="Gambar 2", use_container_width=True)

        if st.button("🔎 Hitung Kemiripan", use_container_width=True):
            try:
                from skimage.metrics import structural_similarity as ssim
                from skimage.metrics import mean_squared_error

                arr1 = np.array(im1)
                arr2 = np.array(im2)
                gray1 = np.mean(arr1, axis=2).astype(np.float32) / 255.0
                gray2 = np.mean(arr2, axis=2).astype(np.float32) / 255.0

                ssim_score = ssim(gray1, gray2, data_range=1.0)
                mse_score = mean_squared_error(gray1, gray2)

                st.success(f"✅ Skor Kemiripan (SSIM): **{ssim_score:.4f}** (semakin mendekati 1 = semakin mirip)")
                st.info(f"📊 Mean Squared Error (MSE): **{mse_score:.6f}** (semakin kecil = semakin mirip)")

                if ssim_score > 0.8:
                    st.balloons()
                    st.markdown("❤️ **Gambar sangat mirip!** Terima kasih telah menggunakan layanan kami. Salam cinta ❤️")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("👆 Unggah dua gambar untuk membandingkannya.")

    # --- KETERANGAN TAMBAHAN DI BAWAH DETEKSI ---
    st.markdown("""
    <div class="footer-note">
        <p>📌 <b>Keterangan:</b> Deteksi kemiripan menggunakan SSIM (Structural Similarity) dan MSE (Mean Squared Error). 
        Semakin tinggi SSIM dan rendah MSE, semakin mirip kedua gambar.</p>
    </div>
    """, unsafe_allow_html=True)
