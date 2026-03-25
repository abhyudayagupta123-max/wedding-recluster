import os
import csv
import shutil
from collections import Counter, defaultdict
from PIL import Image

BASE_DIR = os.path.expanduser("~/wedding-recluster")
CLUSTERS_FILE = os.path.join(BASE_DIR, "all_clusters_rebuilt.csv")
MANIFEST_FILE = os.path.join(BASE_DIR, "face_manifest_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "top31_60_face_crops")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

START_RANK = 30
END_RANK = 60
SAMPLES_PER_CLUSTER = 5
PADDING = 0.35

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

manifest = {}
with open(MANIFEST_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        manifest[row["face_id"]] = row

rows = []
with open(CLUSTERS_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cid = int(row["cluster_id"])
        if cid != -1:
            rows.append(row)

counts = Counter(int(r["cluster_id"]) for r in rows)
ranked = [cid for cid, _ in counts.most_common()]
target_clusters = ranked[START_RANK:END_RANK]

cluster_to_faces = defaultdict(list)
for row in rows:
    cid = int(row["cluster_id"])
    if cid in target_clusters:
        cluster_to_faces[cid].append(row["face_id"])

summary_path = os.path.join(OUT_DIR, "top31_60_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as fsum:
    writer = csv.writer(fsum)
    writer.writerow(["rank", "cluster_id", "cluster_size", "sample_count", "sample_files"])

    for rank, cid in enumerate(target_clusters, start=START_RANK + 1):
        cluster_dir = os.path.join(OUT_DIR, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        copied = []
        chosen = cluster_to_faces[cid][:SAMPLES_PER_CLUSTER]

        for i, face_id in enumerate(chosen, start=1):
            row = manifest.get(face_id)
            if not row:
                continue

            img_path = os.path.join(IMAGE_DIR, row["photo_key"])
            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                left = float(row["left"]) * w
                top = float(row["top"]) * h
                width = float(row["width"]) * w
                height = float(row["height"]) * h

                pad_w = width * PADDING
                pad_h = height * PADDING

                x1 = max(0, int(left - pad_w))
                y1 = max(0, int(top - pad_h))
                x2 = min(w, int(left + width + pad_w))
                y2 = min(h, int(top + height + pad_h))

                crop = img.crop((x1, y1, x2, y2))
                out_name = f"{i:02d}_{face_id[:8]}.jpg"
                out_path = os.path.join(cluster_dir, out_name)
                crop.save(out_path, quality=95)
                copied.append(out_name)
            except Exception:
                continue

        writer.writerow([rank, cid, counts[cid], len(copied), " | ".join(copied)])

print(f"Done. Output folder: {OUT_DIR}")
print(f"Summary file: {summary_path}")
