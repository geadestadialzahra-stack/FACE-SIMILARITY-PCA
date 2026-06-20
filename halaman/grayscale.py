# halaman/grayscale.py - Konversi ke Grayscale
import streamlit as st
from PIL import Image

def tampilkan():
    st.markdown('<h1 class="main-title">🌫️ Konversi ke Grayscale</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload gambar berwarna, lihat hasil hitam-putih</p>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="grayscale_upload")
    if file is not None:
        img = Image.open(file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Gambar Asli (Berwarna)", use_container_width=True)
        with col2:
            gray = img.convert("L")
            st.image(gray, caption="Hasil Grayscale", use_container_width=True)
        st.markdown(f"""
        <div class="explanation-box">
        <b>Informasi:</b><br>
        • Ukuran gambar: {img.size[0]} x {img.size[1]} pixel<br>
        • Mode warna: {img.mode} → Grayscale (L)
        </div>
        """, unsafe_allow_html=True)
