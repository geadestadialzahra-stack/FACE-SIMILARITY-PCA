import streamlit as st
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_people

def tampilkan():
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.pca_model = None
        st.session_state.X_train = None
        st.session_state.show_upload = False
        st.session_state.deteksi_initialized = True
#load sama download data
    if not st.session_state.model_loaded:
        with st.spinner("⏳ Sedang memuat dataset LFW & melatih PCA... Tunggu sebentar yaa!! ^^"):
            try:
                lfw = fetch_lfw_people(
                    min_faces_per_person=5,
                    resize=0.4,
                    color=False,
                    slice_=(slice(50, 200), slice(50, 200))
                )
                X_all = lfw.images
                unique_labels = np.unique(lfw.target)
                valid_labels = [label for label in unique_labels if np.sum(lfw.target == label) >= 5]
                
                if len(valid_labels) < 2:
                    st.error("⚠️ Dataset LFW tidak memiliki cukup orang dengan minimal 5 foto.")
                    st.stop()
                #ambil berapa orang
                selected_labels = valid_labels[:2]
                X_train = []
                for label in selected_labels:
                    idx = np.where(lfw.target == label)[0]
                    for i in idx[:5]:
                        img = lfw.images[i]
                        img_resized = cv2.resize(img, (100, 100))
                        X_train.append(img_resized.flatten() / 255.0)
                
                X_train = np.array(X_train)
                
                # Latih PCA
                k = min(50, len(X_train) - 1)
                pca = PCA(n_components=k)
                pca.fit(X_train)
                
                # Simpan ke session state
                st.session_state.X_train = X_train
                st.session_state.pca_model = pca
                st.session_state.model_loaded = True
                
                st.success(f"✅ Data latih berhasil dimuat! ({len(X_train)} foto dari {len(selected_labels)} orang)")
                
            except Exception as e:
                st.error(f"⚠️ Gagal memuat dataset LFW: {e}")
                st.info("💡 Pastikan koneksi internet aktif. Jika gagal, upload data latih manual.")
                st.session_state.model_loaded = False
#side bar
    with st.sidebar:
        # --- Tombol Sakura ---
        if st.button("🌸", key="toggle_sidebar_deteksi", use_container_width=False):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()

        # --- interaksi tombol ---
        if st.session_state.show_upload:
            st.header("Upload Data Latih Kamu ^^")
            st.markdown("Upload **minimal 10 foto** wajah (2 orang, masing-masing 5+ foto)")
            st.caption("💡 Ini opsional. Tadi kamu sudah download data latih dari internet kok ^^.")

            file_latih = st.file_uploader(
                "Pilih Foto",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="deteksi_train"
            )

            if file_latih:
                st.success(f"✅ {len(file_latih)} foto berhasil terupload!")
            else:
                st.warning("⬆️ Upload foto di sini (opsional)")

            st.header("Atur Ambang Batas Kemiripan Juga!!")
            ambang = st.slider("Atur batas kemiripan", 0.0, 1.0, 0.70, 0.05, key="threshold_deteksi")
            st.caption(f"Threshold saat ini: {ambang:.2f}")

        # --- Penjelasan halaman ---
        st.markdown("""
        <div>
            <h4 style="color: #AD1457; margin-top: 0;">HAII! ^^.</h4>
            <h4 style="color: #AD1457; margin-top: 0;">Ini adalah halaman Deteksi Kemiripan Wajah.</h4>
            <p style="color: #6A1B4D; font-size: 14px; line-height: 1.6;">
                Di sini kamu bisa membandingkan dua foto wajah untuk melihat apakah kedua orang tersebut 
                <b>mirip</b> atau <b>tidak mirip</b>.
            </p>
            <h5 style="color: #AD1457; margin-top: 10px;">Cara Menggunakannya:</h5>
            <ul style="color: #6A1B4D; font-size: 13px; line-height: 1.8; padding-left: 18px;">
                <li><b>1.</b> Kamu bisa langsung upload foto yang mau kamu analisis kok di halaman utama ini ^^</li>
                <li><b>2.</b> (Opsional) Klik <b>"🌸"</b> di atas untuk upload data latih sendiri.</li>
                <li><b>3.</b> Atur threshold. Ini opsional, kaya KKM gitu, mau naruh dimana batas 'mirip'.</li>
                <li><b>4.</b> Klik <b>"Proses Deteksi"</b> untuk melihat hasil.</li>
            </ul>
            <p style="color: #6A1B4D; font-size: 12px; margin-top: 8px; background: #FCE4EC; padding: 6px 12px; border-radius: 6px;">
                💡 <b>Tips:</b> Pastikan foto wajah terlihat jelas, tidak menggunakan filter, tidak menggunakan aksesori yang menutupi wajah, dan tidak berekspresi terlalu berlebihan.
            </p>
        </div>
        """, unsafe_allow_html=True)
#up foto uji
    st.markdown("<h2 style='text-align: center; color: #AD1457; margin-bottom: 50px;'>🔍 Upload Dua Wajah untuk Dibandingkan</h2>", unsafe_allow_html=True)

    kolom1, kolom2 = st.columns(2)
    with kolom1:
        st.markdown("### 📸 Foto Pertama")
        file1 = st.file_uploader("Upload Foto 1", type=["jpg","jpeg","png"], key="f1_deteksi", label_visibility="collapsed")
    with kolom2:
        st.markdown("### 📸 Foto Kedua")
        file2 = st.file_uploader("Upload Foto 2", type=["jpg","jpeg","png"], key="f2_deteksi", label_visibility="collapsed")
#kkm
    if "threshold_deteksi" in st.session_state:
        ambang = st.session_state.threshold_deteksi
    else:
        ambang = 0.70
#proses
    if st.button("🚀 Proses Deteksi Sekarang", use_container_width=True):
        # --- Cek data latih ---
        if not st.session_state.model_loaded:
            st.error("⚠️ **Model PCA belum siap!** Pastikan koneksi internet aktif.")
            st.stop()
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

                # ----- PROSES FOTO UJI -----
                UKURAN = (100, 100)
                pca = st.session_state.pca_model

                v1, _ = praproses(file1.getvalue(), UKURAN)
                v2, _ = praproses(file2.getvalue(), UKURAN)
                img1_warna = muat_warna(file1.getvalue(), UKURAN)
                img2_warna = muat_warna(file2.getvalue(), UKURAN)

                proj1 = pca.transform([v1])
                proj2 = pca.transform([v2])
                kemiripan = cosine_similarity(proj1, proj2)[0][0]

                # ----- TAMPILKAN HASIL -----
                st.subheader("Hasil Deteksi Foto Kamu ^^")
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

                # ----- GRAFIK + PENJELASAN -----
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
                    <b>🟢 Garis hijau titik-titik</b> → <b>Threshold</b> (batas kemiripan) yang kamu atur di sidebar.
                    
                    <br><br>
                    
                    <b>💡 Cara baca:</b><br>
                    Dari 10.000 pixel wajah, PCA bisa meringkasnya menjadi 50 angka saja tanpa kehilangan banyak informasi. Semakin tinggi garis biru, semakin baik representasi wajahnya.
                    </div>
                    """, unsafe_allow_html=True)
