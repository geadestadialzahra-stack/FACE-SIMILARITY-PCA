import streamlit as st
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import os
import requests
from urllib.parse import urlparse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.datasets import fetch_lfw_people
import tempfile
import zipfile
import cv2
import warnings
warnings.filterwarnings("ignore")

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
            width: 100% !important;  /* PAKSA FULL WIDTH */
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

        /* ----- GAYA KHUSUS HASIL DETEKSI ----- */
        .result-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(173,20,87,0.1);
            border: 2px solid #EC407A;
            margin-top: 1.5rem;
            text-align: center;
        }
        .result-card .score {
            font-size: 3rem;
            font-weight: bold;
            color: #AD1457;
        }
        .result-card .label {
            font-size: 1.5rem;
            font-weight: bold;
            color: #D81B60;
        }
        .result-card .detail {
            font-size: 1rem;
            color: #6A1B4D;
            margin-top: 0.5rem;
        }

        /* ----- METRIK KOMPRESI ----- */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }
        .metric-item {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #F8BBD0;
            box-shadow: 0 2px 8px rgba(173,20,87,0.08);
        }
        .metric-item .label {
            font-size: 0.85rem;
            color: #880E4F;
        }
        .metric-item .value {
            font-size: 1.6rem;
            font-weight: bold;
            color: #AD1457;
        }
        .detail-comp {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid #F8BBD0;
        }
        /* ----- BADGE PINK UNTUK LABEL FOTO ----- */
        .pink-badge {
            display: block !important;
            width: 100% !important;
            background: linear-gradient(135deg, #FCE4EC, #F8BBD0) !important;
            color: #AD1457 !important;
            padding: 10px 18px !important;
            border-radius: 12px !important;
            font-weight: bold !important;
            font-size: 16px !important;
            border: 1px solid #EC407A !important;
            box-shadow: 0 2px 10px rgba(233, 30, 99, 0.12) !important;
            text-align: center !important;
            margin-bottom: 12px !important;
        }
        .result-container {
            text-align: center !important;
        }
        .explanation-box {
            background: rgba(255, 255, 255, 0.5) !important;
            padding: 15px !important;
            border-radius: 12px !important;
            border-left: 4px solid #EC407A !important;
            box-shadow: 0 2px 10px rgba(233, 30, 99, 0.08) !important;
            color: #6A1B4D !important;
        }
        .explanation-box b {
            color: #AD1457 !important;
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
st.sidebar.markdown("""
<div class="sidebar-header">
    <span class="logo">🌸</span>
    <span class="title">ANGEL</span>
    <div class="subtitle">✨ Edit & Kreasikan Gambarmu ✨</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align: center; font-size: 14px; color: #880E4F; padding: 0 5px 8px 5px; font-style: italic;">
    Lupakan dia yang membuatmu terluka,<br>semoga web ini bisa membuatmu bahagia. <br> <br> Silahkan pilih menu yang diinginkan !
</div>
""", unsafe_allow_html=True)

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
            if page_name == "🏠 Home":
                st.session_state.home_visited = False
            elif page_name == "🌫️ Grayscale":
                st.session_state.grayscale_visited = False
            elif page_name == "🗜️ Kompresi":
                st.session_state.kompresi_visited = False
            elif page_name == "🔍 Deteksi":
                st.session_state.deteksi_visited = False
            st.rerun()

st.sidebar.markdown("---")
if st.session_state.page == "🏠 Home":
    st.sidebar.markdown('<p class="sidebar-caption">🏠 Home</p>', unsafe_allow_html=True)
elif st.session_state.page == "🌫️ Grayscale":
    st.sidebar.markdown('<p class="sidebar-caption">🌫️ Grayscale</p>', unsafe_allow_html=True)
elif st.session_state.page == "🗜️ Kompresi":
    st.sidebar.markdown('<p class="sidebar-caption">🗜️ Kompresi</p>', unsafe_allow_html=True)
elif st.session_state.page == "🔍 Deteksi":
    st.sidebar.markdown('<p class="sidebar-caption">🔍 Deteksi Kemiripan</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
st.sidebar.markdown("### 👥 Pengembang Aplikasi")
st.sidebar.markdown("**Sarjana Teknik Informatika**")

anggota = [
    {
        "nama": "Gea Destadia Al-Zahra",
        "ig": "@gea_destadia_10",
        "telp": "0831-5068-7481",
    },
    {
        "nama": "Luna Amilia",
        "ig": "@luunaaamiiii",
        "telp": "lunaamilia0@gmail.com",
    },
    {
        "nama": "Nadia Azizah",
        "ig": "@ndyyzh",
        "telp": "0858-4631-3309",
    },
    {
        "nama": "Dalilah Arifah Ariandi DJR",
        "ig": "@adellianav",
        "telp": "0813-1211-6787",
    },
]

for member in anggota:
    st.sidebar.markdown(f"""
    <div class="profile-item">
        <div class="profile-info">
            <div class="name">• {member['nama']} •</div>
            <div class="detail">📸 {member['ig']}</div>
            <div class="detail">📞 {member['telp']}</div>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-university">🎓 Universitas Negeri Semarang</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-footer">
    🌸 Made with Love by Team ANGEL 🌸
</div>
""", unsafe_allow_html=True)


# ======================== HALAMAN UTAMA ========================
page = st.session_state.page

if page == "🏠 Home":
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
        <p>📌 <b>Keterangan:</b> Halaman ini adalah pintu masuk utama. 
        Gunakan menu di sidebar untuk mengakses fitur pengolahan gambar. 
        Selamat berkarya! 🌸</p>
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

elif page == "🌫️ Grayscale":
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
        <p>📌 <b>Keterangan:</b> Fitur ini mengubah gambar berwarna menjadi hitam-putih (grayscale). 
        Hasilnya dapat diunduh dalam format PNG. Cocok untuk efek klasik dan penghematan ukuran file.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Unggah gambar (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="grayscale_uploader"
    )

    # Inisialisasi session state
    if "grayscale_done" not in st.session_state:
        st.session_state.grayscale_done = False
    if "gray_image" not in st.session_state:
        st.session_state.gray_image = None
    if "orig_image" not in st.session_state:
        st.session_state.orig_image = None

    # Reset jika file baru
    if uploaded_file is not None:
        if st.session_state.orig_image is None:
            st.session_state.orig_image = Image.open(uploaded_file)
            st.session_state.grayscale_done = False
            st.session_state.gray_image = None

    if uploaded_file is not None:
        # Tombol konversi
        if st.button("🔄 Konversi ke Grayscale", use_container_width=True):
            if st.session_state.orig_image is not None:
                gray = st.session_state.orig_image.convert("L").convert("RGB")
                st.session_state.gray_image = gray
                st.session_state.grayscale_done = True
                st.rerun()

        # Tampilkan hasil jika sudah diproses
        if st.session_state.grayscale_done and st.session_state.gray_image is not None:
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<div class="pink-badge">🖼️ Gambar Asli</div>', unsafe_allow_html=True)
                st.image(st.session_state.orig_image, use_container_width=True)
                st.caption(f"Ukuran: {st.session_state.orig_image.width} x {st.session_state.orig_image.height} px")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<div class="pink-badge">⚫ Hasil Grayscale</div>', unsafe_allow_html=True)
                st.image(st.session_state.gray_image, use_container_width=True)
                st.caption(f"Ukuran: {st.session_state.gray_image.width} x {st.session_state.gray_image.height} px")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- DI LUAR KOLOM (FULL WIDTH / KIRI) ---
            # Tombol download
            buf = io.BytesIO()
            st.session_state.gray_image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
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
        </div>
        """, unsafe_allow_html=True)

elif page == "🗜️ Kompresi":
    if not st.session_state.kompresi_visited:
        st.balloons()
        st.session_state.kompresi_visited = True

    st.markdown("""
    <div class="kompresi-header">
        <div class="cloud-shower">☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️</div>
        <h1>🗜️ Kompresi Gambar dengan PCA (RGB)</h1>
        <p>Kecilkan ukuran, pertahankan esensi warna.</p>
        <div class="cloud-shower">☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️ ☁️</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #90CAF9; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#0D47A1;">
            ☁️ <b>Principal Component Analysis (PCA)</b> diterapkan pada setiap kanal warna (R, G, B) secara terpisah. 
            Dengan memilih jumlah komponen <b>k</b>, kita dapat mengontrol tingkat kompresi sambil mempertahankan warna asli.
        </p>
        <p style="color:#1565C0; font-style:italic;">
            "Warna adalah jiwa gambar – kompresi tanpa menghilangkan keindahannya."
        </p>
        <p>📌 <b>Keterangan:</b> Kompresi PCA diterapkan pada setiap kanal warna (R, G, B) secara terpisah. 
        Atur jumlah komponen (k) atau persentase varians yang diinginkan. Metrik kualitas (SSIM, PSNR) dan kurva akumulasi membantu mengevaluasi hasil.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Unggah gambar untuk dikompresi (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image, dtype=np.float32)
        h, w, c = img_array.shape  # c = 3

        mode = st.radio(
            "Pilih mode pengaturan kompresi:",
            ["Jumlah komponen (k)", "Persentase varians"],
            horizontal=True,
            key="kompresi_mode"
        )

        max_k = min(h, w)

        if mode == "Jumlah komponen (k)":
            k = st.slider(
                "Jumlah komponen PCA (k) – semakin kecil, semakin besar kompresi",
                min_value=1,
                max_value=max_k,
                value=min(100, max_k),
                step=1,
                key="k_slider"
            )
            variance_target = None
        else:
            variance_target = st.slider(
                "Persentase varians yang dipertahankan (%)",
                min_value=50,
                max_value=100,
                value=95,
                step=1,
                key="variance_slider"
            ) / 100.0
            k = None

        if st.button("🚀 Kompresi dengan PCA", use_container_width=True):
            try:
                if mode == "Persentase varians":
                    pca_full = PCA()
                    pca_full.fit(img_array[:, :, 0])
                    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                    k = np.searchsorted(cumsum, variance_target) + 1
                    if k > max_k:
                        k = max_k
                    st.info(f"Untuk mempertahankan {variance_target*100:.0f}% varians, diperlukan k = {k} komponen.")

                if k > max_k:
                    k = max_k
                    st.warning(f"k dibatasi hingga {max_k} karena dimensi gambar.")

                channels_recon = []
                for i in range(3):
                    channel = img_array[:, :, i]
                    pca = PCA(n_components=k)
                    reduced = pca.fit_transform(channel)
                    recon = pca.inverse_transform(reduced)
                    channels_recon.append(recon)

                reconstructed = np.stack(channels_recon, axis=2)
                reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
                img_reconstructed = Image.fromarray(reconstructed, mode='RGB')

                img_norm = img_array / 255.0
                recon_norm = reconstructed / 255.0
                ssim_val = ssim(img_norm, recon_norm, channel_axis=2, data_range=1.0)
                psnr_vals = []
                for i in range(3):
                    psnr_vals.append(psnr(img_norm[:, :, i], recon_norm[:, :, i], data_range=1.0))
                psnr_val = np.mean(psnr_vals)

                ukuran_asli = h * w * 3
                ukuran_baru = (h * k + k * w) * 3
                rasio = ukuran_baru / ukuran_asli
                penghematan = (1 - rasio) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">🖼️ Gambar Asli (RGB)</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    st.caption(f"Ukuran: {w} x {h} px")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="pink-badge">🗜️ Hasil Kompresi (k={k})</div>', unsafe_allow_html=True)
                    st.image(img_reconstructed, use_container_width=True)
                    st.caption(f"Ukuran: {w} x {h} px")
                    st.markdown('</div>', unsafe_allow_html=True)

                buf = io.BytesIO()
                img_reconstructed.save(buf, format="PNG")
                byte_im = buf.getvalue()
                b64 = base64.b64encode(byte_im).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="compressed_pca.png" style="text-decoration:none;">'
                href += '<button class="download-btn">⬇️ Download Hasil Kompresi</button></a>'
                st.markdown(href, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📊 Metrik Kualitas")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("SSIM", f"{ssim_val:.4f}")
                col_m2.metric("PSNR", f"{psnr_val:.2f} dB")
                col_m3.metric("Penghematan", f"{penghematan:.1f}%")
                col_m4.metric("Rasio Kompresi", f"{rasio:.4f}")

                st.markdown("### 📋 Detail Kompresi")
                st.markdown(f"""
                <div class="detail-comp">
                    <ul>
                        <li><b>Ukuran asli:</b> {ukuran_asli} pixel (3 channel)</li>
                        <li><b>Ukuran setelah PCA (approx):</b> {ukuran_baru} koefisien</li>
                        <li><b>Rasio kompresi:</b> {rasio:.4f}</li>
                        <li><b>Jumlah komponen PCA (per channel):</b> {k}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📝 Kesimpulan Kualitas Kompresi")
                if ssim_val > 0.95 and penghematan > 30:
                    kesimpulan = "✅ **Kompresi sangat baik!** Gambar terkompresi memiliki kualitas hampir sama dengan asli (SSIM > 0.95) dengan penghematan ukuran yang signifikan (>30%)."
                elif ssim_val > 0.85 and penghematan > 20:
                    kesimpulan = "👍 **Kompresi baik.** Kualitas visual masih sangat terjaga (SSIM > 0.85) dengan penghematan ukuran yang cukup (>20%)."
                elif ssim_val > 0.70:
                    kesimpulan = "⚠️ **Kompresi cukup.** Kualitas visual masih dapat diterima (SSIM > 0.70), namun beberapa detail mungkin hilang. Pertimbangkan menaikkan k untuk kualitas lebih baik."
                else:
                    kesimpulan = "❌ **Kompresi kurang baik.** Kualitas visual menurun signifikan (SSIM ≤ 0.70). Sebaiknya naikkan jumlah komponen (k) untuk hasil lebih baik."
                st.markdown(f'<div class="info-box">{kesimpulan}</div>', unsafe_allow_html=True)

                st.markdown("""
                <div style="background: #FCE4EC; padding: 1rem; border-radius: 12px; margin-top: 1rem; border: 1px solid #EC407A;">
                    <p style="margin:0;"><b>💡 Interpretasi Metrik:</b><br>
                    • <b>SSIM</b> – mendekati 1 berarti sangat mirip dengan asli.<br>
                    • <b>PSNR</b> – > 40 dB biasanya kualitas sangat baik.<br>
                    • <b>Penghematan</b> – persentase pengurangan ukuran.<br>
                    • <b>Rasio kompresi</b> – < 1 berarti ukuran berkurang.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📈 Kurva Akumulasi Informasi PCA (Channel R)")
                pca_full = PCA()
                pca_full.fit(img_array[:, :, 0])
                cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-', linewidth=2)
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
                ax.axvline(x=k, color='g', linestyle='--', alpha=0.5, label=f'k = {k}')
                ax.set_xlabel('Jumlah Komponen')
                ax.set_ylabel('Akumulasi Varians')
                ax.set_title('Kurva Akumulasi Informasi PCA (Channel Red)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.balloons()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <p style="font-size:1.2rem; color:#6A1B4D;">👆 Unggah gambar untuk memulai kompresi</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Deteksi":
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Wajah</h1>
        <p>Bandingkan dua wajah dengan metode PCA (Eigenfaces) dan Cosine Similarity.</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Cara kerja:</b> PCA mengekstrak fitur utama (eigenfaces) dari data latih (wajah). 
            Dua wajah yang dibandingkan diproyeksikan ke ruang PCA, lalu dihitung kemiripannya dengan <b>Cosine Similarity</b>.
            Semakin tinggi skor, semakin mirip kedua wajah.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Setiap wajah unik, tapi kecocokan bisa ditemukan."
        </p>
        <p>📌 <b>Keterangan:</b> Deteksi kemiripan menggunakan PCA (Eigenfaces) dan Cosine Similarity. 
        Upload data latih (ZIP) untuk hasil lebih akurat, atau biarkan sistem menggunakan data latih default LFW.</p>
    </div>
    """, unsafe_allow_html=True)

    if "deteksi_model_loaded" not in st.session_state:
        st.session_state.deteksi_model_loaded = False
        st.session_state.deteksi_pca_model = None
        st.session_state.deteksi_X_train = None

    if not st.session_state.deteksi_model_loaded:
        with st.spinner("⏳ Memuat dataset LFW untuk data latih default... Tunggu yaa ^^"):
            try:
                lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.4, color=False)
                unique_labels = np.unique(lfw.target)
                valid_labels = [label for label in unique_labels if np.sum(lfw.target == label) >= 5]
                if len(valid_labels) >= 2:
                    selected = valid_labels[:2]
                    X_train = []
                    for label in selected:
                        idx = np.where(lfw.target == label)[0][:5]
                        for i in idx:
                            img = cv2.resize(lfw.images[i], (100, 100)).flatten() / 255.0
                            X_train.append(img)
                    X_train = np.array(X_train)
                    k = min(50, len(X_train)-1)
                    pca = PCA(n_components=k)
                    pca.fit(X_train)
                    st.session_state.deteksi_pca_model = pca
                    st.session_state.deteksi_X_train = X_train
                    st.session_state.deteksi_model_loaded = True
                    st.success(f"✅ Data latih default LFW dimuat ({len(X_train)} foto dari 2 orang)")
                else:
                    st.warning("⚠️ Dataset LFW tidak mencukupi (minimal 2 orang dengan 5 foto). Upload data latih sendiri.")
            except Exception as e:
                st.warning(f"Gagal memuat LFW: {e}. Upload data latih sendiri.")

    st.markdown("---")
    st.markdown("#### 📂 Data Latih")
    data_mode = st.radio(
        "Pilih sumber data latih:",
        ["Gunakan data latih default (LFW)", "Upload file ZIP berisi gambar wajah"],
        horizontal=True,
        key="data_mode"
    )

    uploaded_zip = None
    if data_mode == "Upload file ZIP berisi gambar wajah":
        uploaded_zip = st.file_uploader("Unggah file ZIP", type=["zip"], key="train_zip_deteksi")
        if uploaded_zip is not None:
            st.success("✅ File ZIP berhasil diunggah.")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Foto Pertama", type=["jpg", "jpeg", "png"], key="img1_deteksi")
    with col_upload2:
        img2 = st.file_uploader("📤 Foto Kedua", type=["jpg", "jpeg", "png"], key="img2_deteksi")

    col_param1, col_param2 = st.columns(2)
    with col_param1:
        n_components = st.slider("Jumlah komponen PCA (k)", 2, 50, 9, 1, key="n_comp_deteksi")
    with col_param2:
        threshold = st.slider("Threshold kemiripan (%)", 0, 100, 70, 5, key="thresh_deteksi") / 100.0

    if img1 is not None and img2 is not None:
        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(img1, caption="Foto Pertama", use_container_width=True)
        with col_show2:
            st.image(img2, caption="Foto Kedua", use_container_width=True)

        if st.button("🔎 Hitung Kemiripan", use_container_width=True):
            try:
                size = (100, 100)
                im1 = Image.open(img1).convert("L").resize(size)
                im2 = Image.open(img2).convert("L").resize(size)
                arr1 = np.array(im1, dtype=np.float32).flatten() / 255.0
                arr2 = np.array(im2, dtype=np.float32).flatten() / 255.0

                train_vectors = None
                if data_mode == "Gunakan data latih default (LFW)" and st.session_state.deteksi_model_loaded:
                    train_vectors = st.session_state.deteksi_X_train
                    pca = st.session_state.deteksi_pca_model
                elif data_mode == "Upload file ZIP berisi gambar wajah" and uploaded_zip is not None:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                        train_vectors = []
                        for root, _, files in os.walk(tmpdir):
                            for file in files:
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        img_path = os.path.join(root, file)
                                        img = Image.open(img_path).convert("L").resize(size)
                                        vec = np.array(img, dtype=np.float32).flatten() / 255.0
                                        train_vectors.append(vec)
                                    except:
                                        continue
                        if len(train_vectors) < 2:
                            st.error("Data latih dari ZIP kurang dari 2 gambar. Gagal melatih PCA.")
                            st.stop()
                        train_vectors = np.array(train_vectors)
                        k = min(n_components, len(train_vectors)-1, len(train_vectors[0]))
                        pca = PCA(n_components=k)
                        pca.fit(train_vectors)
                else:
                    st.error("Tidak ada data latih yang valid. Pilih sumber data latih atau upload ZIP.")
                    st.stop()

                vec1_pca = pca.transform([arr1])[0]
                vec2_pca = pca.transform([arr2])[0]
                sim = cosine_similarity([vec1_pca], [vec2_pca])[0][0]
                kemiripan = sim
                var_ratio = pca.explained_variance_ratio_.sum() * 100
                ambang = threshold

                st.subheader("Hasil Deteksi Foto Kamu ^^")
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama</div>', unsafe_allow_html=True)
                    st.image(img1, caption="Foto Asli", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua</div>', unsafe_allow_html=True)
                    st.image(img2, caption="Foto Asli", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">Skor Kemiripan Foto!!</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{kemiripan:.2%}</h1>", unsafe_allow_html=True)
                    if kemiripan >= ambang:
                        st.success("**WAH MIRIP!! :D**")
                        st.balloons()
                    elif kemiripan >= 0.50:
                        st.warning("**HMM CUKUP MIRIP LAH YA ;D**")
                    else:
                        st.error("**TIDAK MIRIP ^^**")
                    st.caption(f"Komponen PCA: {pca.n_components_}")
                    st.caption(f"Varians: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                kolom_graf, kolom_exp = st.columns([1, 1])
                with kolom_graf:
                    st.subheader("Grafik Akumulasi Informasi")
                    varians = np.cumsum(pca.explained_variance_ratio_)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.plot(range(1, len(varians)+1), varians, 'bo-', linewidth=2, markersize=5)
                    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Varians')
                    ax.axhline(y=ambang, color='green', linestyle=':', linewidth=2, label=f'Threshold {ambang:.2f}')
                    ax.set_xlabel('Jumlah Komponen PCA (k)', fontsize=10)
                    ax.set_ylabel('Akumulasi Informasi', fontsize=10)
                    ax.set_title('Kurva Akumulasi Informasi PCA', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower right', fontsize=8)
                    ax.set_ylim(0, 1.05)
                    st.pyplot(fig)
                with kolom_exp:
                    st.subheader("Penjelasan Grafik!!")
                    st.markdown("""
                    <div class="explanation-box">
                    Grafik ini menunjukkan seberapa banyak <b>informasi wajah</b> yang bisa dipertahankan jika kita menggunakan sejumlah komponen PCA (k).
                    <br><br>
                    <b>🔵 Garis biru</b> → kurva akumulasi varians. Semakin tinggi, semakin baik.<br>
                    <b>🔴 Garis merah putus-putus</b> → 95% varians data sudah terwakili.<br>
                    <b>🟢 Garis hijau titik-titik</b> → <b>Threshold</b> (batas kemiripan) yang kamu atur.
                    <br><br>
                    <b>💡 Cara baca:</b><br>
                    Dari 10.000 pixel wajah, PCA bisa meringkasnya menjadi 50 angka saja tanpa kehilangan banyak informasi. Semakin tinggi garis biru, semakin baik representasi wajahnya.
                    </div>
                    """, unsafe_allow_html=True)

                st.balloons()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <p style="font-size:1.2rem; color:#6A1B4D;">👆 Upload dua foto wajah untuk membandingkan.</p>
        </div>
        """, unsafe_allow_html=True)
