import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os

# Konfigurasi
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Deteksi Manual", layout="centered")
st.title("📸 Deteksi Wajah Manual")

# Emoji emosi
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐"
}

# Upload atau ambil gambar dari kamera
img_file = st.camera_input("🎥 Ambil Foto Wajah")

if img_file is not None:
    # Konversi ke array
    image = Image.open(img_file)
    img_array = np.array(image)

    st.image(img_array, caption="📷 Gambar dari Kamera", use_container_width=True)

    if st.button("📊 Analisis Wajah"):
        with st.spinner("🔍 Menganalisis wajah..."):
            try:
                result = DeepFace.analyze(
                    img_array,
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

                # Tampilkan
                st.success("✅ Wajah berhasil dianalisis!")
                st.markdown(f"**🧓 Umur:** {age}")
                st.markdown(f"**🚻 Gender:** {gender_label} ({gender_scores[predicted_gender]:.2f}%)")
                st.markdown(f"**😶 Emosi:** {emotion} {emoji}")

            except Exception as e:
                st.error(f"❌ Deteksi gagal: {e}")
else:
    st.info("📌 Silakan ambil gambar terlebih dahulu menggunakan kamera di atas.")
