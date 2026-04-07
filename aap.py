import gradio as gr
import numpy as np
import cv2, os, gdown
from tensorflow.keras.models import load_model

# ── Config ──
N, SZ = 15, 96
MODEL_PATH = "best_model.h5"

CLASSES = ["ApplyEyeMakeup","ApplyLipstick","Archery","BabyCrawling","BalanceBeam",
           "BandMarching","BaseballPitch","Basketball","BasketballDunk","BenchPress",
           "Biking","Billiards","BlowDryHair","BlowingCandles","BodyWeightSquats"]

# ── Load Model ──
if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1KfZZYOLqHliU0XNlsxDSsBZU-FKmVpNu", MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)

# ── Frame Extraction ──
def extract_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total-1, N).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (SZ, SZ))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)

    cap.release()

    while len(frames) < N:
        frames.append(np.zeros((SZ, SZ, 3)))

    return np.array(frames)

# ── Prediction ──
def predict(video):
    if video is None:
        return ["No Input"] + ["-"] * 6

    frames = extract_frames(video)
    pred = model(np.expand_dims(frames, 0))[0]

    top5_idx = np.argsort(pred)[-5:][::-1]
    top5 = [(CLASSES[i], f"{pred[i]*100:.2f}%") for i in top5_idx]

    primary = top5[0][0]
    confidence = top5[0][1]

    results = [f"{name} ({conf})" for name, conf in top5]

    return [primary, confidence] + results

# ── CSS ──
css = """
body {
    background:#0b0f19;
    color:#e5e7eb;
    font-family:'Segoe UI', sans-serif;
}
/* HEADINGS */
h2 { color:#f97316; }
p { color:#9ca3af; }
/* CARDS */
.gr-box, .gr-block {
    background:#111827 !important;
    border-radius:12px !important;
    border:1px solid #2a2d36 !important;
}
/* BUTTON */
button {
    background:#f97316 !important;
    color:white !important;
    border:none !important;
    border-radius:8px !important;
    font-weight:600 !important;
    height:45px !important;
}
button:hover {
    background:#ea580c !important;
}
/* SHARE BOX */
.share-box {
    background:#111827;
    padding:12px;
    border-radius:10px;
    margin-top:10px;
    color:#f97316;
    border:1px solid #2a2d36;
}
/* 🔥 REMOVE ORANGE STRIP */
.gradio-container .wrap > div:first-child {
    display: none !important;
}
/* CLEAN VIDEO BOX */
.gradio-container video {
    background:#111827 !important;
}
/* REMOVE ICONS BELOW */
.gradio-container .controls {
    display:none !important;
}
/* ROUND CORNERS FIX */
.gradio-container .wrap {
    border-radius:12px !important;
    overflow:hidden !important;
}
"""

# ── UI ──
with gr.Blocks(css=css) as demo:

    gr.Markdown("## 🎥 VisionAct AI")
    gr.Markdown("Upload video to detect human activity (AI-based)")

    with gr.Row():

        # LEFT
        with gr.Column():

            gr.Markdown("### 📤 Upload Video")

            video_input = gr.Video(height=260)

        # RIGHT
        with gr.Column():

            primary = gr.Textbox(label="🎯 Primary Action")
            confidence = gr.Textbox(label="📊 Confidence Score")

            out1 = gr.Textbox(label="Top 1")
            out2 = gr.Textbox(label="Top 2")
            out3 = gr.Textbox(label="Top 3")
            out4 = gr.Textbox(label="Top 4")
            out5 = gr.Textbox(label="Top 5")

            gr.HTML("""
            <div class='share-box'>
                🔗 Share via Link
            </div>
            """)

    predict_btn = gr.Button("🚀 Analyze Video")

    predict_btn.click(
        fn=predict,
        inputs=video_input,
        outputs=[primary, confidence, out1, out2, out3, out4, out5]
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:20px;color:#6b7280;">
        Built by <b>Arun Kushwah</b> · VisionAct AI
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
