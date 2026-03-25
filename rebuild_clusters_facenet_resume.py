import os
import csv
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import torch
from facenet_pytorch import InceptionResnetV1

# ---------- config ----------
BASE_DIR = os.path.expanduser("~/wedding-recluster")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MANIFEST_FILE = os.path.join(BASE_DIR, "face_manifest_clean.csv")

EMB_FILE = os.path.join(BASE_DIR, "encodings.npy")
META_FILE = os.path.join(BASE_DIR, "encoding_meta.pkl")
STATE_FILE = os.path.join(BASE_DIR, "encoding_state.pkl")
OUT_FILE = os.path.join(BASE_DIR, "all_clusters_rebuilt.csv")

EPS = 0.4
MIN_SAMPLES = 5
SAVE_EVERY = 200
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def save_progress(encodings, face_ids, photo_keys, done_face_ids):
    arr = np.array(encodings, dtype=np.float32)
    np.save(EMB_FILE, arr)
    with open(META_FILE, "wb") as f:
        pickle.dump({"face_ids": face_ids, "photo_keys": photo_keys}, f)
    with open(STATE_FILE, "wb") as f:
        pickle.dump({"done_face_ids": done_face_ids}, f)

def load_existing():
    encodings, face_ids, photo_keys, done_face_ids = [], [], [], set()

    if os.path.exists(EMB_FILE) and os.path.exists(META_FILE):
        arr = np.load(EMB_FILE)
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        encodings = [x for x in arr]
        face_ids = meta["face_ids"]
        photo_keys = meta["photo_keys"]
        done_face_ids = set(face_ids)

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        done_face_ids = set(state.get("done_face_ids", done_face_ids))

    return encodings, face_ids, photo_keys, done_face_ids

def load_face(img_path, row):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    top = int(float(row["top"]) * h)
    left = int(float(row["left"]) * w)
    width = int(float(row["width"]) * w)
    height = int(float(row["height"]) * h)

    right = left + width
    bottom = top + height

    # clamp safely
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    if right <= left or bottom <= top:
        return None

    face = img.crop((left, top, right, bottom)).resize((160, 160))
    arr = np.array(face).astype(np.float32) / 255.0
    ten = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten

def encode_faces():
    encodings, face_ids, photo_keys, done_face_ids = load_existing()

    with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    remaining = [r for r in rows if r["face_id"] not in done_face_ids]
    print(f"Already encoded: {len(done_face_ids)}")
    print(f"Remaining to encode: {len(remaining)}")

    processed_since_save = 0

    for row in tqdm(remaining, total=len(remaining), desc="Encoding faces"):
        face_id = row["face_id"]
        img_path = os.path.join(IMAGE_DIR, row["photo_key"])

        if not os.path.exists(img_path):
            done_face_ids.add(face_id)
            processed_since_save += 1
            if processed_since_save >= SAVE_EVERY:
                save_progress(encodings, face_ids, photo_keys, done_face_ids)
                processed_since_save = 0
            continue

        try:
            face = load_face(img_path, row)
            if face is not None:
                with torch.no_grad():
                    emb = model(face).cpu().numpy()[0]
                encodings.append(emb)
                face_ids.append(face_id)
                photo_keys.append(row["photo_key"])

            done_face_ids.add(face_id)
            processed_since_save += 1

            if processed_since_save >= SAVE_EVERY:
                save_progress(encodings, face_ids, photo_keys, done_face_ids)
                processed_since_save = 0

        except Exception:
            done_face_ids.add(face_id)
            processed_since_save += 1
            if processed_since_save >= SAVE_EVERY:
                save_progress(encodings, face_ids, photo_keys, done_face_ids)
                processed_since_save = 0
            continue

    save_progress(encodings, face_ids, photo_keys, done_face_ids)
    print(f"Encoding complete. Encoded embeddings saved: {len(face_ids)}")

def cluster_only(eps=EPS, min_samples=MIN_SAMPLES):
    if not os.path.exists(EMB_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("Saved embeddings not found. Run encode first.")

    X = np.load(EMB_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    face_ids = meta["face_ids"]
    photo_keys = meta["photo_keys"]

    print(f"Loaded embeddings: {len(face_ids)}")
    print(f"Clustering with eps={eps}, min_samples={min_samples} ...")

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X).labels_

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "face_id", "photo_key"])
        for lbl, fid, pk in zip(labels, face_ids, photo_keys):
            writer.writerow([int(lbl), fid, pk])

    print(f"Saved clusters to: {OUT_FILE}")

if __name__ == "__main__":
    import sys

    mode = "full"
    eps = EPS

    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    if len(sys.argv) >= 3:
        eps = float(sys.argv[2])

    if mode == "encode":
        encode_faces()
    elif mode == "cluster":
        cluster_only(eps=eps)
    else:
        encode_faces()
        cluster_only(eps=eps)
