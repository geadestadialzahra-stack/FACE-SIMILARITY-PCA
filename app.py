# ======================== HALAMAN DETEKSI (PERBAIKAN) ========================
elif page == "🔍 Deteksi":
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Wajah</h1>
        <p>Bandingkan dua wajah dengan PCA (Eigenfaces) + Cosine Similarity</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Cara kerja:</b> PCA mengekstrak fitur utama (eigenfaces) dari data latih 
            (wajah-wajah dari banyak orang). Dua wajah yang dibandingkan diproyeksikan ke ruang PCA, 
            lalu dihitung kemiripannya dengan <b>Cosine Similarity</b>.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Setiap wajah unik, tapi kecocokan bisa ditemukan dengan representasi yang tepat."
        </p>
        <p>📌 <b>Keterangan:</b> Data latih default berasal dari dataset LFW (Labeled Faces in the Wild). 
        Untuk hasil lebih akurat, upload sendiri kumpulan gambar wajah (ZIP).</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Inisialisasi session state untuk model ---
    if "deteksi_model_loaded" not in st.session_state:
        st.session_state.deteksi_model_loaded = False
        st.session_state.deteksi_pca_model = None
        st.session_state.deteksi_scaler = None   # untuk standard scaling
        st.session_state.deteksi_X_train = None

    # --- Load data latih default (LFW) dengan lebih banyak sampel ---
    if not st.session_state.deteksi_model_loaded:
        with st.spinner("⏳ Memuat dataset LFW untuk data latih default... (mohon tunggu)"):
            try:
                # Ambil dataset LFW dengan minimal 5 gambar per orang
                lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.4, color=False)
                unique_labels = np.unique(lfw.target)
                
                # Pilih orang yang memiliki >= 5 gambar, ambil 5 gambar pertama per orang
                # Maksimal 10 orang (total 50 sampel) agar cepat
                max_people = 10
                selected_people = []
                for label in unique_labels:
                    if np.sum(lfw.target == label) >= 5:
                        selected_people.append(label)
                    if len(selected_people) >= max_people:
                        break
                
                if len(selected_people) < 2:
                    st.warning("⚠️ Dataset LFW tidak mencukupi (minimal 2 orang dengan 5 foto). Upload data latih sendiri.")
                    st.session_state.deteksi_model_loaded = False
                else:
                    X_train = []
                    for label in selected_people:
                        idx = np.where(lfw.target == label)[0][:5]  # ambil 5 gambar pertama
                        for i in idx:
                            # Resize ke ukuran tetap (100x100) dan flatten
                            img = cv2.resize(lfw.images[i], (100, 100)).flatten()
                            X_train.append(img)
                    X_train = np.array(X_train, dtype=np.float64)
                    
                    # --- Standardisasi (mean=0, std=1) ---
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    # --- Tentukan k berdasarkan persentase varians (default 95%) ---
                    # Kita akan gunakan PCA tanpa menentukan n_components dulu untuk melihat varians
                    pca_temp = PCA()
                    pca_temp.fit(X_train_scaled)
                    cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                    # Cari k yang mencapai 95% varians
                    target_var = 0.95
                    k_opt = np.searchsorted(cumsum, target_var) + 1
                    # Batasi k agar tidak melebihi jumlah sampel - 1
                    k_opt = min(k_opt, len(X_train_scaled)-1, 100)  # maks 100 komponen
                    
                    # Latih PCA final dengan k_opt komponen
                    pca = PCA(n_components=k_opt)
                    pca.fit(X_train_scaled)
                    
                    # Simpan di session state
                    st.session_state.deteksi_pca_model = pca
                    st.session_state.deteksi_scaler = scaler
                    st.session_state.deteksi_X_train = X_train_scaled
                    st.session_state.deteksi_model_loaded = True
                    st.success(f"✅ Data latih default LFW dimuat: {len(X_train_scaled)} gambar dari {len(selected_people)} orang. "
                               f"Komponen PCA otomatis: {k_opt} (mempertahankan {target_var*100:.0f}% varians)")
            except Exception as e:
                st.warning(f"Gagal memuat LFW: {e}. Upload data latih sendiri.")
                st.session_state.deteksi_model_loaded = False

    # --- Pilihan data latih ---
    st.markdown("---")
    st.markdown("#### 📂 Data Latih")
    data_mode = st.radio(
        "Pilih sumber data latih:",
        ["Gunakan data latih default (LFW - lebih banyak sampel)", "Upload file ZIP berisi gambar wajah"],
        horizontal=True,
        key="data_mode_deteksi"
    )

    uploaded_zip = None
    if data_mode == "Upload file ZIP berisi gambar wajah":
        uploaded_zip = st.file_uploader("Unggah file ZIP (berisi gambar .jpg/.png)", type=["zip"], key="train_zip_deteksi")
        if uploaded_zip is not None:
            st.success("✅ File ZIP berhasil diunggah.")

    # --- Upload dua gambar yang akan dibandingkan ---
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Foto Pertama", type=["jpg", "jpeg", "png"], key="img1_deteksi")
    with col_upload2:
        img2 = st.file_uploader("📤 Foto Kedua", type=["jpg", "jpeg", "png"], key="img2_deteksi")

    # --- Parameter (hanya threshold dan opsi varians) ---
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        # Ganti slider k dengan persentase varians
        var_percent = st.slider(
            "Persentase varians yang dipertahankan (%)",
            min_value=80, max_value=99, value=95, step=1,
            key="var_percent_deteksi"
        ) / 100.0
    with col_param2:
        threshold = st.slider("Threshold kemiripan (%)", 0, 100, 70, 5, key="thresh_deteksi") / 100.0

    # --- Tombol proses ---
    if img1 is not None and img2 is not None:
        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(img1, caption="Foto Pertama", use_container_width=True)
        with col_show2:
            st.image(img2, caption="Foto Kedua", use_container_width=True)

        if st.button("🔎 Hitung Kemiripan", use_container_width=True):
            try:
                size = (100, 100)
                # Baca dan preprocessing gambar uji
                im1 = Image.open(img1).convert("L").resize(size)
                im2 = Image.open(img2).convert("L").resize(size)
                arr1 = np.array(im1, dtype=np.float64).flatten()
                arr2 = np.array(im2, dtype=np.float64).flatten()

                # --- Siapkan data latih sesuai pilihan ---
                if data_mode == "Gunakan data latih default (LFW - lebih banyak sampel)" and st.session_state.deteksi_model_loaded:
                    scaler = st.session_state.deteksi_scaler
                    pca = st.session_state.deteksi_pca_model
                    # Transform data uji menggunakan scaler yang sama
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
                                        img = Image.open(img_path).convert("L").resize(size)
                                        vec = np.array(img, dtype=np.float64).flatten()
                                        train_vectors.append(vec)
                                    except:
                                        continue
                        if len(train_vectors) < 2:
                            st.error("Data latih dari ZIP kurang dari 2 gambar. Gagal melatih PCA.")
                            st.stop()
                        train_vectors = np.array(train_vectors)
                        # Standardisasi
                        scaler = StandardScaler()
                        train_scaled = scaler.fit_transform(train_vectors)
                        # Tentukan k berdasarkan persentase varians yang dipilih
                        pca_temp = PCA()
                        pca_temp.fit(train_scaled)
                        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                        k_opt = np.searchsorted(cumsum, var_percent) + 1
                        k_opt = min(k_opt, len(train_scaled)-1)
                        pca = PCA(n_components=k_opt)
                        pca.fit(train_scaled)
                        # Transform data uji
                        arr1_scaled = scaler.transform([arr1])[0]
                        arr2_scaled = scaler.transform([arr2])[0]
                else:
                    st.error("Tidak ada data latih yang valid. Pilih sumber data latih atau upload ZIP.")
                    st.stop()

                # --- Proyeksi ke ruang PCA ---
                vec1_pca = pca.transform([arr1_scaled])[0]
                vec2_pca = pca.transform([arr2_scaled])[0]

                # --- Normalisasi vektor PCA (L2 norm) untuk cosine similarity ---
                norm1 = np.linalg.norm(vec1_pca)
                norm2 = np.linalg.norm(vec2_pca)
                if norm1 == 0 or norm2 == 0:
                    sim = 0.0
                else:
                    sim = np.dot(vec1_pca, vec2_pca) / (norm1 * norm2)
                kemiripan = sim  # antara -1 dan 1, kita clamp ke [0,1] untuk tampilan
                kemiripan = max(0, min(1, kemiripan))

                # Informasi tambahan
                var_ratio = np.sum(pca.explained_variance_ratio_) * 100
                ambang = threshold

                # --- Tampilkan hasil ---
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
                    st.caption(f"Varians: {var_ratio:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- Grafik akumulasi informasi dan penjelasan ---
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
                    st.subheader("Penjelasan Grafik")
                    st.markdown("""
                    <div class="explanation-box">
                    Grafik ini menunjukkan seberapa banyak <b>informasi wajah</b> yang bisa dipertahankan 
                    dengan sejumlah komponen PCA (k). <br><br>
                    <b>🔵 Garis biru</b> → kurva akumulasi varians. Semakin tinggi, semakin baik.<br>
                    <b>🔴 Garis merah putus-putus</b> → persentase varians yang Anda pilih.<br>
                    <b>🟢 Garis hijau titik-titik</b> → <b>Threshold</b> (batas kemiripan).
                    <br><br>
                    Dengan data latih yang lebih banyak dan representatif, PCA dapat menangkap 
                    variasi antar individu secara lebih baik, sehingga hasil kemiripan lebih akurat.
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
