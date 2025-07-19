import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os


# Tambahan untuk webcam live
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Konfigurasi
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Deteksi Wajah Manual", layout="centered")
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

# === Webcam dengan Streamlit WebRTC === #
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame.to_ndarray(format="bgr24")

# Tampilkan kamera
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

ctx = webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION
)


# Jika kamera aktif dan frame tersedia
if ctx.video_transformer and ctx.video_transformer.frame is not None:
    st.markdown("✅ Kamera aktif. Klik tombol di bawah untuk mengambil snapshot dan analisis.")

    if st.button("📸 Ambil & Analisis Wajah"):
        frame = ctx.video_transformer.frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_array = np.array(image)

        st.image(img_array, caption="📷 Gambar dari Kamera", use_container_width=True)

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

                # Tampilkan hasil
                st.success("✅ Wajah berhasil dianalisis!")
                st.markdown(f"**🧓 Umur:** {age}")
                st.markdown(f"**🚻 Gender:** {gender_label} ({gender_scores[predicted_gender]:.2f}%)")
                st.markdown(f"**😶 Emosi:** {emotion} {emoji}")

            except Exception as e:
                st.error(f"❌ Deteksi gagal: {e}")
else:
    st.info("📌 Nyalakan kamera dan izinkan akses untuk melanjutkan deteksi.")
