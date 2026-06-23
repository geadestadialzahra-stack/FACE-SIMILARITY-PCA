# ======================== BAGIAN DETEKSI (diperbaiki) ========================
elif page == "🔍 Deteksi":
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Wajah</h1>
        <p>Bandingkan dua wajah dengan PCA + Cosine Similarity (dengan normalisasi L2).</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Cara kerja:</b> Deteksi wajah dengan <b>Haar Cascade</b>, crop, resize ke 100x100, 
            histogram equalization, lalu ekstraksi fitur menggunakan <b>PCA</b> yang dilatih pada dataset wajah 
            (LFW dengan banyak orang). Vektor fitur dinormalisasi L2, lalu dihitung <b>Cosine Similarity</b>.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Setiap wajah unik – hasil ini hanya perkiraan, bukan identifikasi mutlak."
        </p>
        <p>📌 <b>Keterangan:</b> Pastikan gambar wajah jelas dan menghadap depan. 
        Jika salah satu tidak terdeteksi wajah, proses tidak dilanjutkan.</p>
    </div>
    """, unsafe_allow_html=True)

    # ========== INISIALISASI DATA LATIH DEFAULT ==========
    if "deteksi_model_loaded" not in st.session_state:
        st.session_state.deteksi_model_loaded = False
        st.session_state.deteksi_pca_model = None
        st.session_state.deteksi_X_train = None

    @st.cache_data
    def load_default_training():
        try:
            # Ambil dataset LFW dengan minimal 20 wajah per orang (agar variatif)
            lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4, color=False)
            X = lfw.images  # shape (n_samples, h, w)
            # Batasi maksimal 500 sampel agar tidak terlalu berat
            if len(X) > 500:
                idx = np.random.choice(len(X), 500, replace=False)
                X = X[idx]
            X_resized = []
            for img in X:
                img_resized = cv2.resize(img, (100, 100)).flatten() / 255.0
                X_resized.append(img_resized)
            X_train = np.array(X_resized)
            if len(X_train) < 10:
                return None, None
            # Gunakan 50 komponen PCA (cukup untuk generalisasi)
            n_comp = min(50, len(X_train)-1)
            pca = PCA(n_components=n_comp)
            pca.fit(X_train)
            return pca, X_train
        except Exception as e:
            return None, None

    if not st.session_state.deteksi_model_loaded:
        with st.spinner("⏳ Memuat dataset LFW (banyak orang)... Tunggu yaa ^^"):
            pca, X_train = load_default_training()
            if pca is not None:
                st.session_state.deteksi_pca_model = pca
                st.session_state.deteksi_X_train = X_train
                st.session_state.deteksi_model_loaded = True
                st.success(f"✅ Data latih default LFW dimuat ({len(X_train)} gambar, PCA komponen={pca.n_components_})")
            else:
                st.warning("⚠️ Gagal memuat LFW. Upload data latih sendiri (ZIP berisi banyak gambar wajah).")

    # ========== FUNGSI DETEKSI DAN PREPROCESS ==========
    def detect_and_preprocess_face(image_pil):
        """Deteksi wajah, crop, resize, equalize, return vector dan gambar hasil crop."""
        img = np.array(image_pil.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        # Parameter lebih ketat untuk mengurangi false positive
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        # Ambil bounding box terbesar
        (x, y, w, h) = max(faces, key=lambda rect: rect[2]*rect[3])
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (100, 100))
        face_eq = cv2.equalizeHist(face_resized)
        face_vector = face_eq.flatten() / 255.0
        return face_vector, face_eq

    # ========== UI ==========
    st.markdown("---")
    st.markdown("#### 📂 Data Latih (opsional)")
    data_mode = st.radio(
        "Pilih sumber data latih:",
        ["Gunakan data latih default (LFW)", "Upload file ZIP berisi gambar wajah (minimal 20)"],
        horizontal=True,
        key="data_mode_deteksi"
    )

    uploaded_zip = None
    if data_mode == "Upload file ZIP berisi gambar wajah (minimal 20)":
        uploaded_zip = st.file_uploader("Unggah file ZIP", type=["zip"], key="train_zip_deteksi")
        if uploaded_zip is not None:
            st.success("✅ File ZIP berhasil diunggah. Pastikan berisi banyak gambar wajah (≥20).")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Foto Pertama (wajah)", type=["jpg", "jpeg", "png"], key="img1_deteksi")
    with col_upload2:
        img2 = st.file_uploader("📤 Foto Kedua (wajah)", type=["jpg", "jpeg", "png"], key="img2_deteksi")

    col_param1, col_param2 = st.columns(2)
    with col_param1:
        n_components = st.slider("Jumlah komponen PCA (k)", 2, 80, 50, 1, key="n_comp_deteksi")
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
                pil1 = Image.open(img1).convert('RGB')
                pil2 = Image.open(img2).convert('RGB')

                vec1, face1 = detect_and_preprocess_face(pil1)
                vec2, face2 = detect_and_preprocess_face(pil2)

                if vec1 is None:
                    st.error("⚠️ Wajah tidak terdeteksi pada Foto Pertama. Pastikan gambar mengandung wajah yang jelas.")
                    st.stop()
                if vec2 is None:
                    st.error("⚠️ Wajah tidak terdeteksi pada Foto Kedua. Pastikan gambar mengandung wajah yang jelas.")
                    st.stop()

                # ===== Siapkan model PCA (dari default atau ZIP) =====
                train_vectors = None
                pca_model = None

                if data_mode == "Gunakan data latih default (LFW)" and st.session_state.deteksi_model_loaded:
                    train_vectors = st.session_state.deteksi_X_train
                    pca_model = st.session_state.deteksi_pca_model
                elif data_mode == "Upload file ZIP berisi gambar wajah (minimal 20)" and uploaded_zip is not None:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                        train_vecs = []
                        for root, _, files in os.walk(tmpdir):
                            for file in files:
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        img_path = os.path.join(root, file)
                                        pil = Image.open(img_path).convert('RGB')
                                        vec, _ = detect_and_preprocess_face(pil)
                                        if vec is not None:
                                            train_vecs.append(vec)
                                    except:
                                        continue
                        if len(train_vecs) < 20:
                            st.error(f"Data latih dari ZIP hanya {len(train_vecs)} wajah. Minimal 20 wajah untuk pelatihan PCA.")
                            st.stop()
                        train_vectors = np.array(train_vecs)
                        n_comp = min(n_components, len(train_vectors)-1, len(train_vectors[0]))
                        pca_model = PCA(n_components=n_comp)
                        pca_model.fit(train_vectors)
                else:
                    st.error("Tidak ada data latih yang valid. Pilih sumber data latih atau upload ZIP yang berisi banyak wajah.")
                    st.stop()

                # ===== Proyeksi dan normalisasi L2 =====
                vec1_pca = pca_model.transform([vec1])[0]
                vec2_pca = pca_model.transform([vec2])[0]
                # Normalisasi L2 agar cosine similarity hanya bergantung arah vektor
                vec1_pca = vec1_pca / (np.linalg.norm(vec1_pca) + 1e-10)
                vec2_pca = vec2_pca / (np.linalg.norm(vec2_pca) + 1e-10)
                sim = np.dot(vec1_pca, vec2_pca)   # cosine similarity
                sim = max(0, sim)                  # clamp ke positif

                var_ratio = np.sum(pca_model.explained_variance_ratio_) * 100
                ambang = threshold

                # ===== Tampilan Hasil =====
                st.subheader("Hasil Deteksi Kemiripan Wajah")
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama (Wajah)</div>', unsafe_allow_html=True)
                    st.image(face1, caption="Wajah yang diekstrak", use_container_width=True, clamp=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua (Wajah)</div>', unsafe_allow_html=True)
                    st.image(face2, caption="Wajah yang diekstrak", use_container_width=True, clamp=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">🎯 Skor Kemiripan</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{sim:.2%}</h1>", unsafe_allow_html=True)
                    if sim >= ambang:
                        st.success("**✅ MIRIP!** (Kemiripan tinggi)")
                        st.balloons()
                    elif sim >= 0.45:
                        st.warning("**⚠️ CUKUP MIRIP** (Kemiripan sedang)")
                    else:
                        st.error("**❌ TIDAK MIRIP** (Kemiripan rendah)")
                    st.caption(f"Komponen PCA: {pca_model.n_components_}")
                    st.caption(f"Varians total: {var_ratio:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                kolom_graf, kolom_exp = st.columns([1, 1])
                with kolom_graf:
                    st.subheader("Grafik Akumulasi Informasi PCA (Data Latih)")
                    varians = np.cumsum(pca_model.explained_variance_ratio_)
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
                    st.subheader("Penjelasan Hasil")
                    st.markdown(f"""
                    <div class="explanation-box">
                    <b>Skor kemiripan:</b> {sim:.2%} (Cosine Similarity setelah normalisasi L2).<br>
                    <b>Ambang batas:</b> {ambang:.0%} – jika skor ≥ ambang, dianggap <b>MIRIP</b>.<br>
                    <b>Komponen PCA:</b> {pca_model.n_components_} dari {len(train_vectors)} sampel latih.<br>
                    <b>Total varians dipertahankan:</b> {var_ratio:.1f}%.<br><br>
                    <b>💡 Interpretasi umum:</b><br>
                    • Skor > 0.70 → kemungkinan besar orang yang sama (wajah sangat mirip).<br>
                    • Skor 0.50–0.70 → kemungkinan orang berbeda dengan fitur agak mirip.<br>
                    • Skor < 0.50 → jelas orang berbeda (tidak mirip).<br><br>
                    <b>⚠️ Catatan:</b> Hasil ini hanya perkiraan berdasarkan fitur wajah, 
                    bukan identifikasi forensik. Faktor pencahayaan, pose, dan ekspresi dapat memengaruhi skor.
                    </div>
                    """, unsafe_allow_html=True)

                st.balloons()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.exception(e)
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <p style="font-size:1.2rem; color:#6A1B4D;">👆 Upload dua foto wajah untuk membandingkan.</p>
        </div>
        """, unsafe_allow_html=True)
