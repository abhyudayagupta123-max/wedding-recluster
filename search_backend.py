import os
import json
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from collections import defaultdict

BASE_DIR = os.path.expanduser("~/wedding-recluster")

PHOTOS_FILE = os.path.join(BASE_DIR, "cluster_to_photos.json")
LABEL_FILE = os.path.join(BASE_DIR, "cluster_to_label.json")
REP_FILE = os.path.join(BASE_DIR, "cluster_representatives.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

with open(PHOTOS_FILE, "r", encoding="utf-8") as f:
    cluster_to_photos = {int(k): set(v) for k, v in json.load(f).items()}

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    cluster_to_label = {int(k): v for k, v in json.load(f).items()}

with open(REP_FILE, "rb") as f:
    cluster_representatives = pickle.load(f)

# IMPORTANT: one name can map to multiple clusters
name_to_clusters = defaultdict(list)
for cid, name in cluster_to_label.items():
    name_to_clusters[name].append(cid)

def preprocess_face(img: Image.Image):
    img = img.convert("RGB").resize((160, 160))
    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten

def embedding_from_pil(img: Image.Image):
    face = preprocess_face(img)
    with torch.no_grad():
        emb = model(face).cpu().numpy()[0]
    return emb

def get_photos_for_person(name):
    if name not in name_to_clusters:
        return set()

    photos = set()
    for cid in name_to_clusters[name]:
        photos |= cluster_to_photos.get(cid, set())
    return photos

def get_photos_for_people(names):
    photo_sets = []

    for name in names:
        photos = get_photos_for_person(name)
        if not photos:
            return []
        photo_sets.append(photos)

    if not photo_sets:
        return []

    result = set.intersection(*photo_sets)
    return sorted(result)

def l2(a, b):
    return float(np.linalg.norm(a - b))

def find_best_cluster_for_embedding(query_emb):
    best_cluster = None
    best_score = 1e9

    for cid, reps in cluster_representatives.items():
        if not reps:
            continue
        dists = [l2(query_emb, np.array(rep)) for rep in reps]
        score = min(dists)
        if score < best_score:
            best_score = score
            best_cluster = cid

    return best_cluster, best_score

def search_by_uploaded_face(image_path, threshold=0.9):
    img = Image.open(image_path)
    query_emb = embedding_from_pil(img)

    best_cluster, best_score = find_best_cluster_for_embedding(query_emb)

    if best_cluster is None:
        return {
            "matched": False,
            "reason": "No cluster found"
        }

    label = cluster_to_label.get(best_cluster, f"Unlabeled Cluster {best_cluster}")
    photos = sorted(cluster_to_photos.get(best_cluster, []))

    return {
        "matched": best_score <= threshold,
        "cluster_id": best_cluster,
        "label": label,
        "score": best_score,
        "photos": photos[:200]
    }

if __name__ == "__main__":
    print("Search backend ready")
    print(f"Labeled unique people: {len(name_to_clusters)}")
    print(f"Labeled clusters: {len(cluster_to_label)}")
    print(f"Clusters with representatives: {len(cluster_representatives)}")
