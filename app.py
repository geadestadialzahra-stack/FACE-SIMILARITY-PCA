# app.py - VERSI FINAL DENGAN TOMBOL SAKURA & GRADASI PINK
# =====================================================
# APLIKASI DETEKSI KEMIRIPAN WAJAH DENGAN PCA (EIGENFACES)
# =====================================================

import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

# ==========================================
# 1. PENGATURAN HALAMAN & CSS (GRADASI PINK SOFT)
# ==========================================
st.set_page_config(
    page_title="PCA Face Similarity",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM (GRADASI PINK + DEKORASI) ---
st.markdown("""
    <style>
        /* BACKGROUND UTAMA DENGAN GRADASI PINK SOFT */
        .stApp {
            background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E9 50%, #FCE4EC 100%) !important;
        }
        .main > div {
            background: transparent !important;
        }
        
        /* SIDEBAR DENGAN GRADASI PINK */
        .css-1d391kg, .css-12w0qpk {
            background: linear-gradient(180deg, #FCE4EC 0%, #FFF0F5 100%) !important;
            border-right: 2px solid #F8BBD0 !important;
        }
        
        /* DEKORASI BUNGA SAKURA DI BACKGROUND (CSS) */
        .stApp::before {
            content: "🌸 🌸 🌸";
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 24px;
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
            animation: float 6s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-10px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        
        /* JUDUL */
        .main-title {
            text-align: center;
            color: #AD1457;
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 5px;
            text-shadow: 0 2px 10px rgba(173, 20, 87, 0.15);
        }
        .sub-title {
            text-align: center;
            color: #D81B60;
            font-size: 18px;
            margin-bottom: 30px;
        }
        
        /* CARD HASIL */
        .result-card {
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(233, 30, 99, 0.15);
            text-align: center;
            border: 1px solid #F8BBD0;
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(233, 30, 99, 0.2);
        }
        
        /* SEMBUNYIKAN ELEMEN BAWAAN */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* TOMBOL PROSES */
        .stButton button {
            background: linear-gradient(135deg, #EC407A, #D81B60);
            color: white;
            font-size: 18px;
            border-radius: 50px;
            padding: 10px 30px;
            width: 100%;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3);
        }
        .stButton button:hover {
            transform: scale(1.03) translateY(-2px);
            box-shadow: 0 8px 25px rgba(233, 30, 99, 0.4);
        }
        
        /* JUDUL DI SIDEBAR */
        h1, h2, h3 {
            color: #AD1457;
        }
        
        /* LOGO BULAT DENGAN ANIMASI */
        .rounded-logo {
            border-radius: 50%;
            border: 3px solid #EC407A;
            box-shadow: 0 4px 15px rgba(233, 30, 99, 0.25);
            transition: all 0.5s ease;
        }
        .rounded-logo:hover {
            transform: rotate(10deg) scale(1.05);
            box-shadow: 0 8px 30px rgba(233, 30, 99, 0.35);
        }
        
        /* DAFTAR ANGGOTA */
        .member-list {
            color: #6A1B4D;
            font-size: 14px;
            line-height: 1.8;
            background: rgba(255,255,255,0.4);
            padding: 12px;
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }
        
        /* INFO PREPROCESSING */
        .preprocess-info {
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            padding: 15px;
            border-radius: 12px;
            font-size: 14px;
            color: #6A1B4D;
            border-left: 4px solid #EC407A;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(233, 30, 99, 0.1);
        }
        
        /* FILE UPLOADER */
        .stFileUploader {
            background: rgba(255,255,255,0.6) !important;
            border-radius: 12px !important;
            border: 2px dashed #EC407A !important;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        }
        .stFileUploader:hover {
            border-color: #D81B60 !important;
            background: rgba(255,255,255,0.8) !important;
        }
        
        /* TOMBOL SAKURA (TANPA TEKS) */
        .sakura-toggle {
            background: transparent !important;
            border: none !important;
            font-size: 28px !important;
            cursor: pointer !important;
            transition: all 0.5s ease !important;
            padding: 5px 10px !important;
            border-radius: 50% !important;
            line-height: 1 !important;
        }
        .sakura-toggle:hover {
            transform: scale(1.2) rotate(15deg) !important;
            background: rgba(236, 64, 122, 0.15) !important;
            box-shadow: 0 0 20px rgba(236, 64, 122, 0.2) !important;
        }
        
        /* ANIMASI FADE UNTUK UPLOAD SECTION */
        .upload-section {
            animation: fadeSlide 0.5s ease;
        }
        @keyframes fadeSlide {
            0% { opacity: 0; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        /* SLIDER */
        .stSlider > div {
            background: rgba(255,255,255,0.4) !important;
            border-radius: 10px !important;
        }
        
        /* METRIC */
        .stMetric {
            background: rgba(255,255,255,0.3) !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI DETEKSI WAJAH (Haar Cascade)
# ==========================================
def detect_and_crop_face(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_crop = gray[y:y+h, x:x+w]
        return face_crop, True
    else:
        return gray, False

# ==========================================
# 3. FUNGSI PREPROCESSING
# ==========================================
def preprocess_with_face_detection(file_bytes, img_size=(100, 100)):
    face_crop, detected = detect_and_crop_face(file_bytes)
    resized = cv2.resize(face_crop, img_size)
    normalized = resized / 255.0
    vector = normalized.flatten()
    return vector, resized, detected

# ==========================================
# 4. INISIALISASI SESSION STATE
# ==========================================
if "show_upload" not in st.session_state:
    st.session_state.show_upload = True

# ==========================================
# 5. JUDUL (tanpa tombol di sini)
# ==========================================
st.markdown('<p class="main-title">🌸 Deteksi Kemiripan Wajah</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Menggunakan Metode PCA (Eigenfaces) & Cosine Similarity</p>', unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR (dengan tombol sakura di pojok kanan atas)
# ==========================================
with st.sidebar:
    # --- BARIS ATAS: LOGO + TOMBOL SAKURA ---
    col_logo, col_sakura = st.columns([4, 1])
    with col_logo:
        st.markdown(
            '<img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="60" class="rounded-logo">',
            unsafe_allow_html=True
        )
    with col_sakura:
        # Tombol sakura tanpa teks
        sakura_label = "🌸"  # selalu sama
        if st.button(sakura_label, key="toggle_upload_sidebar", use_container_width=False):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()
    
    # --- BAGIAN UPLOAD DATA LATIH (dengan animasi fade) ---
    if st.session_state.show_upload:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📂 Upload Data Latih")
        st.markdown("Upload **minimal 10 foto** wajah. <br> (Usahakan 2 orang berbeda, masing-masing 5+ foto) <br> Pastikan Mukanya Terlihat yaa <br> Dan No Filter yaa ^^", unsafe_allow_html=True)
        
        uploaded_train_files = st.file_uploader(
            "Pilih Foto Buat Latih ^^",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="train"
        )
        
        if uploaded_train_files:
            st.success(f"✅ {len(uploaded_train_files)} yey foto sudah terupload!")
        else:
            st.warning("⬆️ Upload foto di sini")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("🌸 Upload data latih disembunyikan. Klik sakura di atas untuk muncul.")

    st.divider()
    
    # --- SLIDER THRESHOLD (tetap muncul) ---
    threshold = st.slider("🎯 Atur Ambang Batas Kemiripan", 0.0, 1.0, 0.70, 0.05)
    st.caption(f"Threshold saat ini: {threshold:.2f}")
    
    st.divider()
    
    # --- DAFTAR ANGGOTA ---
    st.markdown(
        """
        <div class="member-list">
            <b>🌸 Dibuat oleh Kelompok 2</b><br>
            Anggota:<br>
            1. Gea Destadia Al-Zahra<br>
            2. Luna Amilia<br>
            3. Dalilah Arifah Ariandi DJR<br>
            4. Nadia Azzizah
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 7. AREA UTAMA: UPLOAD 2 FOTO UJI
# ==========================================
st.header("🔍 Upload Dua Wajah yang Mau Kamu Bandingkan ^^")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 📸 Foto Pertama")
    face1_file = st.file_uploader("Upload Foto 1", type=["jpg", "jpeg", "png"], key="face1", label_visibility="collapsed")

with col2:
    st.markdown("#### 📸 Foto Kedua")
    face2_file = st.file_uploader("Upload Foto 2", type=["jpg", "jpeg", "png"], key="face2", label_visibility="collapsed")

# ==========================================
# 8. TOMBOL PROSES
# ==========================================
st.markdown("---")
col_button_proses, _ = st.columns([1, 3])
with col_button_proses:
    proses_button = st.button("🚀 Proses Deteksi Sekarang", use_container_width=True)

# ==========================================
# 9. LOGIKA PEMROSESAN
# ==========================================
if proses_button:
    if not uploaded_train_files or len(uploaded_train_files) < 10:
        st.error("⚠️ **Data Latih Kurang!** Upload minimal 10 gambar wajah (2 orang, masing-masing 5 foto).")
    elif not face1_file or not face2_file:
        st.error("⚠️ **Foto Uji Kurang!** Upload kedua foto yang akan dibandingkan.")
    else:
        with st.spinner("⏳ Sedang memproses... Mohon tunggu."):
            time.sleep(0.5)
            
            # --- A. Preprocessing Data Latih ---
            IMG_SIZE = (100, 100)
            X_train = []
            train_detected_status = []
            
            progress_bar = st.progress(0, text="Mengolah data latih...")
            for i, file in enumerate(uploaded_train_files):
                file_bytes = file.getvalue()
                vector, _, detected = preprocess_with_face_detection(file_bytes, IMG_SIZE)
                X_train.append(vector)
                train_detected_status.append(detected)
                progress_bar.progress((i + 1) / len(uploaded_train_files))
            
            X_train = np.array(X_train)
            
            # --- B. Preprocessing Foto Uji ---
            progress_bar.progress(50, text="Memproses foto uji...")
            
            face1_bytes = face1_file.getvalue()
            vec1, img1_resized, detected1 = preprocess_with_face_detection(face1_bytes, IMG_SIZE)
            
            face2_bytes = face2_file.getvalue()
            vec2, img2_resized, detected2 = preprocess_with_face_detection(face2_bytes, IMG_SIZE)
            
            # --- C. Jalankan PCA ---
            progress_bar.progress(70, text="Menjalankan PCA & mencari Eigenfaces...")
            k = min(50, len(X_train) - 1) if len(X_train) > 1 else 1
            pca = PCA(n_components=k)
            X_pca = pca.fit_transform(X_train)
            
            proj1 = pca.transform([vec1])
            proj2 = pca.transform([vec2])
            
            similarity = cosine_similarity(proj1, proj2)[0][0]
            
            progress_bar.progress(100, text="Selesai!")
            time.sleep(0.3)
            progress_bar.empty()
            
            # ==========================================
            # 10. TAMPILKAN HASIL
            # ==========================================
            st.markdown("---")
            st.subheader("📊 Hasil Deteksi Foto Kamu ^^")
            
            st.markdown(f"""
            <div class="preprocess-info">
                <b>📌 Proses Preprocessing:</b><br>
                • Gambar diubah ke Grayscale ✓<br>
                • Resize ke {IMG_SIZE[0]}x{IMG_SIZE[1]} pixel ✓<br>
                • Normalisasi pixel (0-1) ✓<br>
                • Flatten ke 1 vektor ✓<br>
                • Deteksi wajah: Foto 1 {'Berhasil' if detected1 else '⚠️ Gagal (pakai full gambar)'} | Foto 2 {'✅ Berhasil' if detected2 else '⚠️ Gagal (pakai full gambar)'}
            </div>
            """, unsafe_allow_html=True)
            
            col_res1, col_res2, col_res3 = st.columns(3, gap="medium")
            
            with col_res1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(img1_resized, width=180, caption="Foto 1 (Setelah Resize)", use_container_width=False)
                st.caption(f"Ukuran: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_res2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(img2_resized, width=180, caption="Foto 2 (Setelah Resize)", use_container_width=False)
                st.caption(f"Ukuran: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_res3:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.metric(label="Skor Kemiripan", value=f"{similarity:.2%}", delta=None)
                
                if similarity >= threshold:
                    st.success("**Kesimpulan: MIRIP**")
                    st.markdown("*(Kemungkinan besar orang yang sama)*")
                    st.balloons()
                elif similarity >= 0.50:
                    st.warning("**Kesimpulan: CUKUP MIRIP**")
                    st.markdown("*(Perlu verifikasi tambahan)*")
                else:
                    st.error("**Kesimpulan: TIDAK MIRIP**")
                    st.markdown("*(Kemungkinan besar orang berbeda)*")
                
                st.caption(f"Jumlah Komponen PCA: {k}")
                st.caption(f"Varians data: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                st.caption(f"Threshold yang digunakan: {threshold:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # --- Grafik PCA ---
            st.subheader("Grafik Akumulasi Informasi PCA")
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(explained_variance)+1), explained_variance, 'bo-', linewidth=2, markersize=6)
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Varians')
            ax.axhline(y=threshold, color='green', linestyle=':', linewidth=2, label=f'Threshold: {threshold:.2f}')
            ax.set_xlabel('Jumlah Komponen PCA (k)', fontsize=12)
            ax.set_ylabel('Akumulasi Informasi', fontsize=12)
            ax.set_title('Kurva Akumulasi Informasi PCA', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)