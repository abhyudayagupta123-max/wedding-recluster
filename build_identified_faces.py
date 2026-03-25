import os
import csv
import shutil
from collections import defaultdict
from PIL import Image

BASE_DIR = os.path.expanduser("~/wedding-recluster")
CLUSTERS_FILE = os.path.join(BASE_DIR, "all_clusters_rebuilt.csv")
MANIFEST_FILE = os.path.join(BASE_DIR, "face_manifest_clean.csv")
LABELS_FILE = os.path.join(BASE_DIR, "cluster_labels.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "identified_faces")

os.makedirs(OUT_DIR, exist_ok=True)

# Load labels
labels = {}
with open(LABELS_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["confidence"].lower() == "high":
            labels[int(row["cluster_id"])] = row["person_name"]

# Load manifest
manifest = {}
with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        manifest[row["face_id"]] = row

# Process clusters
count = 0

with open(CLUSTERS_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cid = int(row["cluster_id"])
        if cid not in labels:
            continue

        name = labels[cid]
        face_id = row["face_id"]
        meta = manifest.get(face_id)

        if not meta:
            continue

        img_path = os.path.join(IMAGE_DIR, meta["photo_key"])
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            left = float(meta["left"]) * w
            top = float(meta["top"]) * h
            width = float(meta["width"]) * w
            height = float(meta["height"]) * h

            pad = 0.3
            x1 = max(0, int(left - width * pad))
            y1 = max(0, int(top - height * pad))
            x2 = min(w, int(left + width * (1 + pad)))
            y2 = min(h, int(top + height * (1 + pad)))

            crop = img.crop((x1, y1, x2, y2))

            person_dir = os.path.join(OUT_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            out_path = os.path.join(person_dir, f"{face_id[:8]}.jpg")
            crop.save(out_path, quality=95)

            count += 1

        except:
            continue

print(f"Done. Saved {count} faces into identified_faces/")
