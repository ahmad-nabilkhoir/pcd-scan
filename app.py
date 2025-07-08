import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os

# Konfigurasi lingkungan
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Deteksi Manual", layout="centered")
st.title("📸 Deteksi Wajah Manual")

# Emoji untuk emosi
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐"
}

# Inisialisasi session state
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False
if "img" not in st.session_state:
    st.session_state.img = None

# Tombol untuk mulai kamera
if st.button("🎥 Mulai Kamera"):
    st.session_state.camera_started = True
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        st.error("❌ Kamera tidak tersedia atau tidak bisa dibuka.")
    else:
        ret, frame = cam.read()
        cam.release()

        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.img = img
            st.image(img, channels="RGB", caption="📷 Gambar dari Kamera", use_container_width=True)
        else:
            st.error("❌ Gagal membaca gambar dari kamera.")

# Tombol analisis wajah
if st.session_state.img is not None:
    if st.button("📊 Analisis Wajah"):
        with st.spinner("🔍 Menganalisis wajah..."):
            try:
                result = DeepFace.analyze(
                    st.session_state.img,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=True,
                    detector_backend="mediapipe"
                )

                # Ambil hasil
                age = result[0]['age']
                gender_scores = result[0]['gender']
                emotion = result[0]['dominant_emotion']

                predicted_gender = max(gender_scores, key=gender_scores.get)
                gender_label = "Laki-laki" if predicted_gender == "Man" else "Perempuan"
                emoji = emoji_map.get(emotion.lower(), "❓")

                # Tampilkan hasil analisis
                st.success("✅ Wajah berhasil dianalisis!")
                st.markdown(f"**🧓 Umur:** {age}")
                st.markdown(f"**🚻 Gender:** {gender_label} ({gender_scores[predicted_gender]:.2f}%)")
                st.markdown(f"**😶 Emosi:** {emotion} {emoji}")

            except Exception as e:
                st.error(f"❌ Deteksi gagal: {e}")
else:
    st.info("📌 Klik tombol 'Mulai Kamera' untuk mengambil gambar.")
