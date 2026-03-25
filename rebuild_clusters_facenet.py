import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import torch
from facenet_pytorch import InceptionResnetV1

# ---- setup ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

image_folder = "images"
manifest_file = "face_manifest_clean.csv"

encodings = []
face_ids = []
photo_keys = []

# ---- helper ----
def load_face(img_path, box):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    top = int(float(box["top"]) * h)
    left = int(float(box["left"]) * w)
    width = int(float(box["width"]) * w)
    height = int(float(box["height"]) * h)

    face = img.crop((left, top, left+width, top+height))
    face = face.resize((160, 160))

    face = np.array(face) / 255.0
    face = torch.tensor(face).permute(2,0,1).float().unsqueeze(0).to(device)

    return face

# ---- encoding ----
with open(manifest_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in tqdm(reader):
        img_path = os.path.join(image_folder, row["photo_key"])

        if not os.path.exists(img_path):
            continue

        try:
            face = load_face(img_path, row)
            with torch.no_grad():
                emb = model(face).cpu().numpy()[0]

            encodings.append(emb)
            face_ids.append(row["face_id"])
            photo_keys.append(row["photo_key"])

        except Exception:
            continue

print(f"Total faces encoded: {len(encodings)}")

# ---- clustering ----
print("Clustering...")

X = np.array(encodings)

clustering = DBSCAN(
    eps=0.9,
    min_samples=5,
    metric='euclidean'
).fit(X)

labels = clustering.labels_

# ---- save ----
print("Saving results...")

with open("all_clusters_rebuilt.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["cluster_id", "face_id", "photo_key"])

    for lbl, fid, pk in zip(labels, face_ids, photo_keys):
        writer.writerow([lbl, fid, pk])

print("Done.")
