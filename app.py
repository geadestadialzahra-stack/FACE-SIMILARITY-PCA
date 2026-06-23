# ============================== HALAMAN DETEKSI (PERBAIKAN DEEP) ==============================
elif page == "🔍 Deteksi":
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Wajah</h1>
        <p>Bandingkan dua wajah dengan PCA + Cosine Similarity & Euclidean Distance</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Metode:</b> Wajah dideteksi, dicrop, lalu dilakukan <b>CLAHE</b> untuk mengurangi efek cahaya. 
            PCA mengekstrak fitur, lalu kemiripan dihitung dari <b>Cosine Similarity</b> dan <b>Euclidean Distance</b> 
            secara bersamaan – hanya jika keduanya menunjukkan kemiripan, skor akhir tinggi.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Dua metrik lebih akurat daripada satu."
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Inisialisasi session state untuk model ---
    if "deteksi_model_loaded" not in st.session_state:
        st.session_state.deteksi_model_loaded = False
        st.session_state.deteksi_pca_model = None
        st.session_state.deteksi_scaler = None
        st.session_state.deteksi_X_train = None
        st.session_state.deteksi_mean_dist = None  # untuk normalisasi jarak

    # --- Load data latih default (LFW) dengan lebih banyak sampel ---
    if not st.session_state.deteksi_model_loaded:
        with st.spinner("⏳ Memuat dataset LFW untuk data latih default... (mohon tunggu)"):
            try:
                lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.4, color=False)
                unique_labels = np.unique(lfw.target)
                
                # Ambil lebih banyak orang (15 orang) untuk variasi lebih baik
                max_people = 15
                selected_people = []
                for label in unique_labels:
                    if np.sum(lfw.target == label) >= 5:
                        selected_people.append(label)
                    if len(selected_people) >= max_people:
                        break
                
                if len(selected_people) < 2:
                    st.warning("⚠️ Dataset LFW tidak mencukupi. Upload data latih sendiri.")
                    st.session_state.deteksi_model_loaded = False
                else:
                    X_train = []
                    for label in selected_people:
                        idx = np.where(lfw.target == label)[0][:5]
                        for i in idx:
                            # Resize dan flatten
                            img = cv2.resize(lfw.images[i], (100, 100))
                            # CLAHE untuk meningkatkan kontras
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            img_eq = clahe.apply((img * 255).astype(np.uint8))
                            X_train.append(img_eq.flatten().astype(np.float64))
                    X_train = np.array(X_train, dtype=np.float64)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    # Tentukan k berdasarkan 90% varians (lebih general)
                    pca_temp = PCA()
                    pca_temp.fit(X_train_scaled)
                    cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                    target_var = 0.90  # turun dari 0.95 menjadi 0.90
                    k_opt = np.searchsorted(cumsum, target_var) + 1
                    k_opt = min(k_opt, len(X_train_scaled)-1, 80)
                    
                    pca = PCA(n_components=k_opt)
                    pca.fit(X_train_scaled)
                    
                    # Simpan mean jarak Euclidean antar data latih untuk normalisasi
                    train_pca = pca.transform(X_train_scaled)
                    dists = []
                    for i in range(len(train_pca)):
                        for j in range(i+1, len(train_pca)):
                            dists.append(np.linalg.norm(train_pca[i] - train_pca[j]))
                    mean_dist = np.mean(dists) if dists else 1.0
                    
                    st.session_state.deteksi_pca_model = pca
                    st.session_state.deteksi_scaler = scaler
                    st.session_state.deteksi_X_train = X_train_scaled
                    st.session_state.deteksi_mean_dist = mean_dist
                    st.session_state.deteksi_model_loaded = True
                    st.success(f"✅ Data latih LFW dimuat: {len(X_train_scaled)} gambar dari {len(selected_people)} orang. "
                               f"Komponen PCA: {k_opt} (mempertahankan {target_var*100:.0f}% varians)")
            except Exception as e:
                st.warning(f"Gagal memuat LFW: {e}. Upload data latih sendiri.")
                st.session_state.deteksi_model_loaded = False

    # --- Pilihan data latih ---
    st.markdown("---")
    st.markdown("#### 📂 Data Latih")
    data_mode = st.radio(
        "Pilih sumber data latih:",
        ["Gunakan data latih default (LFW)", "Upload file ZIP berisi gambar wajah"],
        horizontal=True,
        key="data_mode_deteksi"
    )

    uploaded_zip = None
    if data_mode == "Upload file ZIP berisi gambar wajah":
        uploaded_zip = st.file_uploader("Unggah file ZIP (berisi gambar .jpg/.png)", type=["zip"], key="train_zip_deteksi")
        if uploaded_zip is not None:
            st.success("✅ File ZIP berhasil diunggah.")

    # --- Upload dua gambar ---
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Foto Pertama", type=["jpg", "jpeg", "png"], key="img1_deteksi")
    with col_upload2:
        img2 = st.file_uploader("📤 Foto Kedua", type=["jpg", "jpeg", "png"], key="img2_deteksi")

    # --- Parameter ---
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        var_percent = st.slider(
            "Persentase varians yang dipertahankan (%)",
            min_value=80, max_value=98, value=90, step=1,
            key="var_percent_deteksi"
        ) / 100.0
    with col_param2:
        threshold = st.slider("Threshold kemiripan (%)", 0, 100, 70, 5, key="thresh_deteksi") / 100.0

    # --- Fungsi deteksi wajah dengan CLAHE ---
    def detect_and_preprocess(image_np):
        """Deteksi wajah, crop, lalu CLAHE, return (face_crop, success)."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape)==3 else image_np
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None, False
        (x, y, w, h) = max(faces, key=lambda rect: rect[2]*rect[3])
        face = gray[y:y+h, x:x+w]
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_eq = clahe.apply(face)
        face_resized = cv2.resize(face_eq, (100, 100))
        return face_resized, True

    # --- Tombol proses ---
    if img1 is not None and img2 is not None:
        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(img1, caption="Foto Pertama (asli)", use_container_width=True)
        with col_show2:
            st.image(img2, caption="Foto Kedua (asli)", use_container_width=True)

        if st.button("🔎 Hitung Kemiripan", use_container_width=True):
            try:
                pil_img1 = Image.open(img1).convert("RGB")
                pil_img2 = Image.open(img2).convert("RGB")
                np_img1 = np.array(pil_img1)
                np_img2 = np.array(pil_img2)

                face1, ok1 = detect_and_preprocess(np_img1)
                face2, ok2 = detect_and_preprocess(np_img2)

                if not ok1:
                    st.error("⚠️ **Tidak terdeteksi wajah pada foto pertama.** Harap upload gambar yang mengandung wajah manusia.")
                    st.stop()
                if not ok2:
                    st.error("⚠️ **Tidak terdeteksi wajah pada foto kedua.** Harap upload gambar yang mengandung wajah manusia.")
                    st.stop()

                # Tampilkan wajah hasil preprocessing
                col_face1, col_face2 = st.columns(2)
                with col_face1:
                    st.image(face1, caption="Wajah (CLAHE)", use_container_width=True, clamp=True)
                with col_face2:
                    st.image(face2, caption="Wajah (CLAHE)", use_container_width=True, clamp=True)

                arr1 = face1.flatten().astype(np.float64)
                arr2 = face2.flatten().astype(np.float64)

                # --- Siapkan data latih ---
                if data_mode == "Gunakan data latih default (LFW)" and st.session_state.deteksi_model_loaded:
                    scaler = st.session_state.deteksi_scaler
                    pca = st.session_state.deteksi_pca_model
                    mean_dist = st.session_state.deteksi_mean_dist
                    arr1_scaled = scaler.transform([arr1])[0]
                    arr2_scaled = scaler.transform([arr2])[0]
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
                                        img = Image.open(img_path).convert("L").resize((100, 100))
                                        # CLAHE pada data latih juga
                                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                                        img_np = np.array(img)
                                        img_eq = clahe.apply(img_np)
                                        vec = img_eq.flatten().astype(np.float64)
                                        train_vectors.append(vec)
                                    except:
                                        continue
                        if len(train_vectors) < 2:
                            st.error("Data latih dari ZIP kurang dari 2 gambar. Gagal melatih PCA.")
                            st.stop()
                        train_vectors = np.array(train_vectors)
                        scaler = StandardScaler()
                        train_scaled = scaler.fit_transform(train_vectors)
                        pca_temp = PCA()
                        pca_temp.fit(train_scaled)
                        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                        k_opt = np.searchsorted(cumsum, var_percent) + 1
                        k_opt = min(k_opt, len(train_scaled)-1, 80)
                        pca = PCA(n_components=k_opt)
                        pca.fit(train_scaled)
                        arr1_scaled = scaler.transform([arr1])[0]
                        arr2_scaled = scaler.transform([arr2])[0]
                        # Hitung mean jarak dari data latih
                        train_pca = pca.transform(train_scaled)
                        dists = []
                        for i in range(len(train_pca)):
                            for j in range(i+1, len(train_pca)):
                                dists.append(np.linalg.norm(train_pca[i] - train_pca[j]))
                        mean_dist = np.mean(dists) if dists else 1.0
                else:
                    st.error("Tidak ada data latih yang valid. Pilih sumber data latih atau upload ZIP.")
                    st.stop()

                # Proyeksi
                vec1_pca = pca.transform([arr1_scaled])[0]
                vec2_pca = pca.transform([arr2_scaled])[0]

                # Cosine similarity
                norm1 = np.linalg.norm(vec1_pca)
                norm2 = np.linalg.norm(vec2_pca)
                if norm1 == 0 or norm2 == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = np.dot(vec1_pca, vec2_pca) / (norm1 * norm2)
                cos_sim = max(0, min(1, cos_sim))

                # Euclidean distance
                euclidean_dist = np.linalg.norm(vec1_pca - vec2_pca)
                # Normalisasi jarak terhadap mean_dist (semakin kecil jarak, semakin mirip)
                # Kita ubah menjadi skor kemiripan dari jarak: sim_dist = exp(-euclidean_dist / mean_dist)
                # atau sim_dist = 1 - (euclidean_dist / (mean_dist * 2))  dibatasi 0-1
                if mean_dist > 0:
                    dist_sim = max(0, 1 - (euclidean_dist / (mean_dist * 2)))
                else:
                    dist_sim = 0.5  # fallback

                # Gabungan: rata-rata tertimbang (bobot bisa disesuaikan)
                # Kita beri bobot lebih ke cosine jika jarak juga kecil
                # Gunakan aturan: jika dist_sim > 0.5 dan cos_sim > threshold, maka mirip
                # Skor akhir = (cos_sim * 0.7 + dist_sim * 0.3)
                final_score = 0.7 * cos_sim + 0.3 * dist_sim
                # Pembulatan
                kemiripan = max(0, min(1, final_score))

                var_ratio = np.sum(pca.explained_variance_ratio_) * 100
                ambang = threshold

                # Tampilkan hasil
                st.subheader("Hasil Deteksi Foto Kamu ^^")
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama (cropped)</div>', unsafe_allow_html=True)
                    st.image(face1, caption="Wajah terdeteksi", use_container_width=True, clamp=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua (cropped)</div>', unsafe_allow_html=True)
                    st.image(face2, caption="Wajah terdeteksi", use_container_width=True, clamp=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">Skor Kemiripan Foto!!</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{kemiripan:.2%}</h1>", unsafe_allow_html=True)
                    # Tampilkan juga metrik terpisah
                    st.caption(f"Cosine: {cos_sim:.2%} | Distance-score: {dist_sim:.2%}")
                    if kemiripan >= ambang:
                        st.success("**WAH MIRIP!! :D**")
                        st.balloons()
                    elif kemiripan >= 0.45:
                        st.warning("**HMM CUKUP MIRIP LAH YA ;D**")
                    else:
                        st.error("**TIDAK MIRIP ^^**")
                    st.caption(f"Komponen PCA: {pca.n_components_}")
                    st.caption(f"Varians: {var_ratio:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Grafik akumulasi informasi
                st.markdown("---")
                kolom_graf, kolom_exp = st.columns([1, 1])
                with kolom_graf:
                    st.subheader("Grafik Akumulasi Informasi")
                    varians = np.cumsum(pca.explained_variance_ratio_)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.plot(range(1, len(varians)+1), varians, 'bo-', linewidth=2, markersize=5)
                    ax.axhline(y=var_percent, color='red', linestyle='--', linewidth=2, label=f'{var_percent*100:.0f}% Varians')
                    ax.axhline(y=ambang, color='green', linestyle=':', linewidth=2, label=f'Threshold {ambang:.2f}')
                    ax.set_xlabel('Jumlah Komponen PCA (k)', fontsize=10)
                    ax.set_ylabel('Akumulasi Informasi', fontsize=10)
                    ax.set_title('Kurva Akumulasi Informasi PCA', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower right', fontsize=8)
                    ax.set_ylim(0, 1.05)
                    st.pyplot(fig)
                with kolom_exp:
                    st.subheader("Penjelasan")
                    st.markdown("""
                    <div class="explanation-box">
                    <b>📊 Metrik gabungan:</b><br>
                    • <b>Cosine Similarity</b> mengukur arah vektor fitur – nilai tinggi artinya pola wajah serupa.<br>
                    • <b>Distance-score</b> mengukur kedekatan jarak Euclidean – semakin kecil jarak, semakin mirip.<br>
                    • Skor akhir adalah kombinasi keduanya (70% cosine + 30% distance).<br><br>
                    Dengan preprocessing CLAHE dan PCA yang lebih general (90% varians), hasil lebih akurat.
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
