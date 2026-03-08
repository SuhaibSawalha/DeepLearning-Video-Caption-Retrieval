import streamlit as st
import pickle
import numpy as np
import torch
import open_clip
import os

EMBEDDING_FILE = "msrvtt_video_embeddings.pkl"
VIDEO_DIR = "./TrainValVideo"
TOP_K = 5

@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device
    )
    model.eval()
    return model, device

@torch.no_grad()
def encode_text(text, model, device):
    tokens = open_clip.tokenize([text]).to(device)
    feat = model.encode_text(tokens)
    feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]

@st.cache_resource
def load_database():
    with open(EMBEDDING_FILE, "rb") as f:
        db = pickle.load(f)

    video_ids = list(db.keys())
    embeddings = np.stack([db[v] for v in video_ids])
    return video_ids, embeddings

st.set_page_config(page_title="Text-to-Video Retrieval", layout="wide")

st.title("🎥 Text-to-Video Retrieval (MSR-VTT)")
st.markdown("Type a sentence and retrieve the most relevant videos.")

query = st.text_input("🔍 Enter text query")

if query:
    model, device = load_model()
    video_ids, video_embeddings = load_database()

    text_feat = encode_text(query, model, device)
    scores = video_embeddings @ text_feat

    idxs = np.argsort(scores)[::-1][:TOP_K]

    st.subheader("📊 Top Results")

    cols = st.columns(TOP_K)

    for col, idx in zip(cols, idxs):
        vid = video_ids[idx]
        score = scores[idx]
        video_path = os.path.join(VIDEO_DIR, vid + ".mp4")

        with col:
            st.markdown(f"**{vid}**")
            st.markdown(f"Similarity: `{score:.3f}`")

            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning("Video file not found")


# python3 -m streamlit run test.py