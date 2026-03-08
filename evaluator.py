import os
import json
import pickle
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
from typing import Dict, List
import matplotlib.pyplot as plt

import time
from functools import wraps
from contextlib import contextmanager

VIDEO_DIR = "./TrainValVideo"
INFO_JSON = "./caption.json"
EMBEDDINGS_FILE = "msrvtt_video_embeddings.pkl"

NUM_FRAMES = 8
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# =========================
# Timing utilities
# =========================

def _sync_device():
    """Best-effort sync so timings include GPU/MPS work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # MPS doesn't always require/offer explicit sync the same way; this is best-effort.
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass

def timed(fn):
    """Decorator to measure wall-clock time of a function."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_device()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync_device()
        elapsed = time.perf_counter() - start
        print(f"⏱️  {fn.__name__} took {elapsed:.3f} sec")
        return result
    return wrapper

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    _sync_device()
    start = time.perf_counter()
    yield
    _sync_device()
    elapsed = time.perf_counter() - start
    print(f"⏱️  {name} took {elapsed:.3f} sec")

class VectorEngine:
    @timed
    def __init__(self):
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=self.device
        )
        self.model.eval()
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            test_out = self.model.encode_image(dummy)
            self.embed_dim = test_out.shape[-1]
            print(f"Embedding dimension: {self.embed_dim}")

    @torch.no_grad()
    def encode_images(self, images):
        images = images.to(self.device)
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = open_clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")


def sample_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    if total < num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)

    cap.release()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    return frames


@timed
def load_video_captions() -> Dict[str, List[str]]:
    with open(INFO_JSON, "r") as f:
        data = json.load(f)

    captions_per_video = {}
    for vid, info in data.items():
        video_id = "video" + vid if not vid.startswith("video") else vid
        if "captions" in info and len(info["captions"]) > 0:
            captions_per_video[video_id] = info["captions"]
    return captions_per_video


@timed
def build_video_embeddings(video_ids: List[str]):
    engine = VectorEngine()
    db = {}
    failed = []

    for vid in tqdm(video_ids, desc="Encoding videos"):
        path = os.path.join(VIDEO_DIR, f"{vid}.mp4")

        if not os.path.exists(path):
            failed.append((vid, "file not found"))
            continue

        with timer(f"sample_frames({vid})"):
            frames = sample_frames(path, NUM_FRAMES)

        if len(frames) == 0:
            failed.append((vid, "no frames extracted"))
            continue

        try:
            with timer(f"encode_video({vid})"):
                imgs = torch.stack([engine.preprocess(f) for f in frames])
                frame_feats = engine.encode_images(imgs)

                video_feat = frame_feats.mean(axis=0)
                video_feat = video_feat / np.linalg.norm(video_feat)

            if not np.isfinite(video_feat).all():
                failed.append((vid, "invalid embedding"))
                continue

            db[vid] = video_feat

        except Exception as e:
            failed.append((vid, str(e)))
            continue

    if failed:
        print(f"Failed on {len(failed)} videos")

    with timer("Saving embeddings"):
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(db, f)

    print(f"Saved embeddings to {EMBEDDINGS_FILE}")


@timed
def evaluate(captions_per_video: Dict[str, List[str]]):
    engine = VectorEngine()

    with timer("Loading embeddings"):
        with open(EMBEDDINGS_FILE, "rb") as f:
            video_db = pickle.load(f)
    
    video_ids = sorted(list(video_db.keys()))
    video_matrix = np.stack([video_db[v] for v in video_ids])

    all_ranks = []
    per_video_best_ranks = []

    with timer("Retrieval loop"):
        for gt_vid, captions in tqdm(
            captions_per_video.items(),
            desc="Evaluating retrieval"
        ):
            if gt_vid not in video_db:
                continue

            video_ranks = []
            gt_idx = video_ids.index(gt_vid)

            for caption in captions:
                text_feat = engine.encode_text([caption])[0]
                scores = video_matrix @ text_feat
                rank = (scores > scores[gt_idx]).sum() + 1  # 1-based rank
                all_ranks.append(rank)
                video_ranks.append(rank)

            per_video_best_ranks.append(min(video_ranks))

    all_ranks = np.array(all_ranks, dtype=np.int32)
    best_ranks = np.array(per_video_best_ranks, dtype=np.int32)

    # =========================
    # mAP (single-relevant retrieval): AP(query) = 1 / rank
    # =========================
    with timer("mAP computation"):
        map_per_caption = float(np.mean(1.0 / all_ranks)) if len(all_ranks) else float("nan")
        map_best_per_video = float(np.mean(1.0 / best_ranks)) if len(best_ranks) else float("nan")

    print("\n" + "="*60)
    print("📊 MSR-VTT Text-to-Video Retrieval Results")
    print("="*60)
    
    print(f"\n📈 Dataset size: {len(video_ids)} videos, {len(all_ranks)} queries")
    
    print("\n🎯 Per-Caption Metrics (all captions evaluated separately):")
    print(f"  R@1   : {100 * np.mean(all_ranks <= 1):.2f}%")
    print(f"  R@5   : {100 * np.mean(all_ranks <= 5):.2f}%")
    print(f"  R@10  : {100 * np.mean(all_ranks <= 10):.2f}%")
    print(f"  R@50  : {100 * np.mean(all_ranks <= 50):.2f}%")
    print(f"  Median: {np.median(all_ranks):.1f}")
    print(f"  Mean  : {np.mean(all_ranks):.2f}")
    print(f"  mAP   : {map_per_caption:.4f}")
    
    print("\n⭐ Per-Video Best Metrics (best caption per video):")
    print(f"  R@1   : {100 * np.mean(best_ranks <= 1):.2f}%")
    print(f"  R@5   : {100 * np.mean(best_ranks <= 5):.2f}%")
    print(f"  R@10  : {100 * np.mean(best_ranks <= 10):.2f}%")
    print(f"  R@50  : {100 * np.mean(best_ranks <= 50):.2f}%")
    print(f"  Median: {np.median(best_ranks):.1f}")
    print(f"  Mean  : {np.mean(best_ranks):.2f}")
    print(f"  mAP   : {map_best_per_video:.4f}")
    
    random_baseline_r1 = 100.0 / len(video_ids)
    print(f"\nRandom baseline R@1: {random_baseline_r1:.2f}%")
    print("="*60)

    plot_results(all_ranks, len(video_ids), title_suffix="(Per-Caption)")


def plot_results(ranks, num_videos, title_suffix=""):
    ranks = np.asarray(ranks)

    plt.figure(figsize=(7, 5))
    plt.hist(
        ranks,
        bins=50,
        log=True,
        edgecolor="black"
    )
    plt.xlabel("Rank of Correct Video")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Rank Distribution {title_suffix}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    ks = np.arange(1, num_videos + 1)
    recall = np.array([(ranks <= k).mean() for k in ks])

    plt.figure(figsize=(7, 5))
    plt.plot(ks, recall, linewidth=2)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title(f"Recall@K Curve {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    captions_per_video = load_video_captions()
    video_ids = sorted(captions_per_video.keys())

    if not os.path.exists(EMBEDDINGS_FILE):
        build_video_embeddings(video_ids)
    else:
        print(f"Loading existing embeddings from {EMBEDDINGS_FILE}")

    evaluate(captions_per_video)