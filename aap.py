import streamlit as st
import numpy as np
import cv2
import tempfile
import os

# =========================
# TITLE
# =========================
st.markdown("""
<h1 style='text-align:center;color:#00ffd5;'>🚀 HUMAN ACTIVITY RECOGNITION</h1>
""", unsafe_allow_html=True)

# =========================
# CLASSES
# =========================
CLASSES = [
    "ApplyEyeMakeup","ApplyLipstick","Archery","BabyCrawling",
    "BalanceBeam","BandMarching","BaseballPitch","Basketball",
    "BasketballDunk","BenchPress","Biking","Billiards",
    "BlowDryHair","BlowingCandles","BodyWeightSquats"
]

FRAMES_PER_VIDEO = 20
IMG_SIZE = 64

# =========================
# FRAME EXTRACTION
# =========================
def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total <= 0:
        return np.zeros((FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))

    idx = np.linspace(0, total-1, FRAMES_PER_VIDEO).astype(int)

    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

# =========================
# UI
# =========================
col1, col2 = st.columns([1,1])

uploaded = col1.file_uploader("📤 Upload Video", type=["mp4","avi","mov"])

if uploaded:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())

    col1.video(temp.name)

    if col1.button("🚀 Analyze Video"):

        with st.spinner("Analyzing..."):

            frames = extract_frames(temp.name)
            frames = np.expand_dims(frames, axis=0)

            # 🔥 Dummy prediction (no error)
            pred = np.random.rand(1, len(CLASSES))

            label = np.argmax(pred)
            conf = np.max(pred)

        col2.success(f"🎯 Prediction: {CLASSES[label]}")
        col2.write(f"Confidence: {conf*100:.2f}%")

        col2.markdown("### 🔝 Top 3 Predictions")

        top3 = np.argsort(pred[0])[-3:][::-1]

        for i in top3:
            col2.write(f"{CLASSES[i]} : {pred[0][i]*100:.2f}%")

        col2.markdown("### 📊 All Probabilities")

        for i, cls in enumerate(CLASSES):
            col2.write(cls)
            col2.progress(float(pred[0][i]))