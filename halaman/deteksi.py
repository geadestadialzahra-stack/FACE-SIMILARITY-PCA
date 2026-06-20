# halaman/deteksi.py - Halaman Deteksi Kemiripan Wajah
import streamlit as st
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def tampilkan():
    # ==========================================
    # SIDEBAR: UPLOAD DATA LATIH + THRESHOLD
    # ==========================================
    with st.sidebar:
        # --- Tombol Sakura (tanpa garis dan caption) ---
        st.markdown('<div class="sakura-btn-container" style="margin-top: 5px;">', unsafe_allow_html=True)
        kol1, kol2, kol3 = st.columns([1, 2, 1])
        with kol2:
            if st.button("🌸", key="toggle_sidebar_deteksi"):
                st.session_state.show_upload = not st.session_state.show_upload
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # --- BLOK YANG MUNCUL/HILANG BERSAMAAN ---
        if st.session_state.show_upload:
            st.header("📂 Upload Data Latih")
            st.markdown("Upload **minimal 10 foto** wajah (2 orang, masing-masing 5+ foto)")

            file_latih = st.file_uploader(
                "Pilih Foto",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="deteksi_train"
            )

            if file_latih:
                st.success(f"✅ {len(file_latih)} foto berhasil terupload!")
            else:
                st.warning("⬆️ Upload foto di sini")

            st.header("🎯 Ambang Batas Kemiripan")
            ambang = st.slider("Atur batas kemiripan", 0.0, 1.0, 0.70, 0.05, key="threshold_deteksi")
            st.caption(f"Threshold saat ini: {ambang:.2f}")

        else:
            st.info("🌸 Upload dan pengaturan disembunyikan. Klik sakura di atas.")

        # --- Penjelasan halaman (tetap muncul) ---
        st.divider()
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.5); padding: 15px; border-radius: 12px; border-left: 4px solid #EC407A;">
            <h4 style="color: #AD1457; margin-top: 0;">🌸 Halo! Selamat datang di halaman Deteksi Kemiripan Wajah.</h4>
            <p style="color: #6A1B4D; font-size: 14px; line-height: 1.6;">
                Di sini kamu bisa membandingkan dua foto wajah untuk melihat apakah kedua orang tersebut 
                <b>mirip</b> atau <b>tidak mirip</b>.
            </p>
            <h5 style="color: #AD1457; margin-top: 10px;">📌 Cara Menggunakan:</h5>
            <ul style="color: #6A1B4D; font-size: 13px; line-height: 1.8; padding-left: 18px;">
                <li><b>1.</b> Klik <b>"🌸"</b> di atas untuk menampilkan upload data latih & pengaturan.</li>
                <li><b>2.</b> Upload minimal <b>10 foto wajah</b> dari 2 orang berbeda (masing-masing 5 foto).</li>
                <li><b>3.</b> Upload dua foto uji di bagian bawah.</li>
                <li><b>4.</b> Atur threshold (batas kemiripan) dengan slider.</li>
                <li><b>5.</b> Klik <b>"Proses Deteksi"</b> untuk melihat hasil.</li>
            </ul>
            <p style="color: #6A1B4D; font-size: 12px; margin-top: 8px; background: #FCE4EC; padding: 6px 12px; border-radius: 6px;">
                💡 <b>Tips:</b> Pastikan foto wajah terlihat jelas dan tidak menggunakan filter.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # AREA UTAMA: UPLOAD 2 FOTO UJI
    # ==========================================
    st.markdown("<h2 style='text-align: center; color: #AD1457;'>🔍 Upload Dua Wajah untuk Dibandingkan</h2>", unsafe_allow_html=True)

    kolom1, kolom2 = st.columns(2)
    with kolom1:
        st.markdown("### 📸 Foto Pertama")
        file1 = st.file_uploader("Upload Foto 1", type=["jpg","jpeg","png"], key="f1_deteksi", label_visibility="collapsed")
    with kolom2:
        st.markdown("### 📸 Foto Kedua")
        file2 = st.file_uploader("Upload Foto 2", type=["jpg","jpeg","png"], key="f2_deteksi", label_visibility="collapsed")

    # ==========================================
    # TOMBOL PROSES
    # ==========================================
    if st.button("🚀 Proses Deteksi Sekarang", use_container_width=True):
        try:
            train_files = file_latih
        except NameError:
            train_files = None

        if not train_files or len(train_files) < 10:
            st.error("⚠️ **Data Latih Kurang!** Upload minimal 10 foto.")
            st.info("💡 Klik tombol 🌸 di sidebar untuk menampilkan bagian upload.")
        elif not file1 or not file2:
            st.error("⚠️ Upload kedua foto uji!")
        else:
            with st.spinner("⏳ Sedang memproses... Mohon tunggu."):
                time.sleep(0.5)

                # ----- FUNGSI DETEKSI WAJAH -----
                def deteksi_dan_potong_wajah(byte_gambar):
                    arr_np = np.frombuffer(byte_gambar, np.uint8)
                    img = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
                    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    wajah = cascade.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if len(wajah) > 0:
                        x, y, w, h = max(wajah, key=lambda rect: rect[2] * rect[3])
                        return abu[y:y+h, x:x+w], True
                    return abu, False

                def praproses(byte_gambar, ukuran=(100, 100)):
                    potongan, _ = deteksi_dan_potong_wajah(byte_gambar)
                    resize = cv2.resize(potongan, ukuran)
                    normal = resize / 255.0
                    return normal.flatten(), resize

                def muat_warna(byte_gambar, ukuran=(100, 100)):
                    arr_np = np.frombuffer(byte_gambar, np.uint8)
                    img = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
                    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    wajah = cascade.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if len(wajah) > 0:
                        x, y, w, h = max(wajah, key=lambda rect: rect[2] * rect[3])
                        potongan = img[y:y+h, x:x+w]
                        resize = cv2.resize(potongan, ukuran)
                        return cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                    else:
                        resize = cv2.resize(img, ukuran)
                        return cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)

                # ----- PROSES DATA LATIH -----
                UKURAN = (100, 100)
                X_latih = []
                progress = st.progress(0, text="Mengolah data latih...")
                for i, file in enumerate(train_files):
                    vektor, _ = praproses(file.getvalue(), UKURAN)
                    X_latih.append(vektor)
                    progress.progress((i+1)/len(train_files))
                X_latih = np.array(X_latih)

                # ----- PCA -----
                progress.progress(50, text="Menjalankan PCA...")
                k = min(50, len(X_latih)-1) if len(X_latih)>1 else 1
                pca = PCA(n_components=k)
                pca.fit(X_latih)

                # ----- PROSES FOTO UJI -----
                progress.progress(70, text="Memproses foto uji...")
                v1, _ = praproses(file1.getvalue(), UKURAN)
                v2, _ = praproses(file2.getvalue(), UKURAN)
                img1_warna = muat_warna(file1.getvalue(), UKURAN)
                img2_warna = muat_warna(file2.getvalue(), UKURAN)

                proj1 = pca.transform([v1])
                proj2 = pca.transform([v2])
                kemiripan = cosine_similarity(proj1, proj2)[0][0]
                progress.empty()

                # ----- TAMPILKAN HASIL -----
                st.markdown("---")
                st.subheader("📊 Hasil Deteksi")
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama</div>', unsafe_allow_html=True)
                    st.image(img1_warna, caption=f"Resize {UKURAN[0]}x{UKURAN[1]}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua</div>', unsafe_allow_html=True)
                    st.image(img2_warna, caption=f"Resize {UKURAN[0]}x{UKURAN[1]}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">🎯 Skor Kemiripan</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{kemiripan:.2%}</h1>", unsafe_allow_html=True)
                    if kemiripan >= ambang:
                        st.success("✅ **MIRIP**")
                        st.balloons()
                    elif kemiripan >= 0.50:
                        st.warning("⚠️ **CUKUP MIRIP**")
                    else:
                        st.error("❌ **TIDAK MIRIP**")
                    st.caption(f"Komponen PCA: {k}")
                    st.caption(f"Varians: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # ----- GRAFIK -----
                st.subheader("📈 Grafik Akumulasi Informasi")
                varians = np.cumsum(pca.explained_variance_ratio_)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(varians)+1), varians, 'bo-', linewidth=2)
                ax.axhline(y=0.95, color='r', linestyle='--', label='95% Varians')
                ax.axhline(y=ambang, color='g', linestyle=':', label=f'Threshold {ambang:.2f}')
                ax.set_xlabel('Jumlah Komponen (k)')
                ax.set_ylabel('Akumulasi Varians')
                ax.set_title('Kurva Akumulasi Informasi PCA')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
