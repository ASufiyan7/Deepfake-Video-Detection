import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import numpy as np
import os
import tempfile
from PIL import Image

from model import (
    FullDeepFakeDetector,
    compute_fft_batch,
)

st.set_page_config(page_title="MarkV Deepfake Detector (Simple XAI)", layout="wide")

# Model loading with caching
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "MarkV_weights.pth"

    if not os.path.exists(model_path):
        st.error("Missing MarkV_weights.pth file.")
        return None, None, None

    model = FullDeepFakeDetector(base_weights_path=None, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    mtcnn = MTCNN(image_size=224, device=device, post_process=False)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, mtcnn, transform


model, mtcnn, transform = load_models()

# Video face extraction and preprocessing
def process_video_for_analysis(video_file, mtcnn, transform, frame_skip=10):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        vid_path = tfile.name

    cap = cv2.VideoCapture(vid_path)
    frames, crops, frame_ids = [], [], []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

            if boxes is not None:
                faces = mtcnn.extract(rgb, boxes, save_path=None)

                if faces is not None:
                    face_pil = transforms.ToPILImage()(faces[0].byte())
                    crops.append(face_pil)

                    face = faces[0] / 255.0
                    if face.ndim == 2:
                        face = face.unsqueeze(0).repeat(3, 1, 1)
                    elif face.shape[0] == 1:
                        face = face.repeat(3, 1, 1)

                    face = transform(face).to(next(model.parameters()).device)
                    frames.append(face)
                    frame_ids.append(idx)

        idx += 1

    cap.release()
    try:
        os.remove(vid_path)
    except Exception:
        pass

    return frames, crops, frame_ids


# Per-frame scoring 
def get_frame_scores(frame_model, frames):
    scores_real = []

    with torch.no_grad():
        for face in frames:
            batch = face.unsqueeze(0)
            try:
                freq = compute_fft_batch(batch)
                feat = frame_model(batch, freq)
            except Exception:
                feat = frame_model(batch)

            seq = feat.unsqueeze(1)
            logit = model.temporal(seq)
            score_real = float(torch.sigmoid(logit).item())
            scores_real.append(score_real)

    return scores_real


# Simple temporal influence 
def compute_simple_influence(scores):
    arr = np.array(scores, dtype=np.float32)
    raw = np.abs(arr - 0.5)
    total = raw.sum() + 1e-8
    if total == 0:
        return np.ones_like(raw) / max(1, raw.size)
    return (raw / total).tolist()


def simple_explanation(is_video_fake, frame_score, influence):
    fake_score = 1.0 - frame_score
    if fake_score > 0.75:
        decision = "Model is highly confident this frame is manipulated."
    elif fake_score > 0.55:
        decision = "Model sees clear signs of manipulation in this frame."
    elif frame_score > 0.75:
        decision = "Model is highly confident this frame looks authentic."
    elif frame_score > 0.55:
        decision = "Model finds this frame mostly consistent with a real face."
    else:
        decision = "This frame lies near the model's decision boundary."

    influence_pct = influence * 100.0
    if influence > 0.25:
        inf_s = f"This frame is one of the most influential frames (~{influence_pct:.1f}% of per-frame influence)."
    elif influence > 0.10:
        inf_s = f"This frame has a moderate influence (~{influence_pct:.1f}%)."
    else:
        inf_s = f"This frame has a small influence (~{influence_pct:.1f}%)."

    if is_video_fake:
        reason = "(Selected because it pushes the model towards a FAKE decision.)"
    else:
        reason = "(Selected because it pushes the model towards a REAL decision.)"

    return f"{decision} {inf_s} {reason}"


# Streamlit UI
st.title("MarkV Deepfake Detector — Simple XAI 🔍")

if model is None:
    st.stop()

uploaded = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

if uploaded:
    st.video(uploaded.getvalue())

    if st.button("Analyze Video"):
        with st.spinner("Detecting faces and extracting frames..."):
            frames, crops, frame_ids = process_video_for_analysis(uploaded, mtcnn, transform)

        if not frames:
            st.error("No faces detected in the video.")
            st.stop()

        # Video-level decision using up to 10 frames
        with st.spinner("Running temporal model for video-level decision..."):
            seq = frames.copy()
            if len(seq) < 10:
                seq += [seq[-1]] * (10 - len(seq))
            seq_tensor = torch.stack(seq[:10]).unsqueeze(0)
            with torch.no_grad():
                video_logit = model(seq_tensor).item()
                real_prob = float(torch.sigmoid(torch.tensor(video_logit)).item())
            fake_prob = 1.0 - real_prob
            is_fake = fake_prob > 0.5

        # Per-frame scores
        with st.spinner("Scoring frames..."):
            real_scores = get_frame_scores(model.frame_model, frames)

        temporal_influence = compute_simple_influence(real_scores)

        # Verdict
        if is_fake:
            st.error(f"FAKE detected — Confidence: {fake_prob:.1%}")
        else:
            st.success(f"REAL video — Confidence: {real_prob:.1%}")

        st.divider()

        top_k = 5
        scores_np = np.array(real_scores)
        abs_dist = np.abs(scores_np - 0.5)

        if is_fake:
            fake_scores = 1.0 - scores_np
            combined = fake_scores * 100.0 + abs_dist
            top_idx = np.argsort(-combined)[:top_k]
        else:
            combined = scores_np * 100.0 + abs_dist
            top_idx = np.argsort(-combined)[:top_k]

        top_idx = [int(i) for i in top_idx]

        st.subheader("Top frames that influenced the decision")
        cols = st.columns(len(top_idx))

        for i, frame_idx in enumerate(top_idx):
            with cols[i]:
                st.caption(f"Frame #{frame_ids[frame_idx]}")
                st.image(crops[frame_idx], use_container_width=True)

                frame_score = real_scores[frame_idx]
                st.write(f"Real score: **{frame_score*100:.1f}%**")

                influence = temporal_influence[frame_idx]
                expl = simple_explanation(is_fake, frame_score, influence)
                st.write("**Why this frame?**")
                st.write(expl)

        st.info("Notes: Explanations are intentionally short and simple — they summarise model confidence and relative influence of the frame on the final decision.")
