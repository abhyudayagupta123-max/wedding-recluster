import os
import csv
import shutil
from collections import Counter, defaultdict

BASE_DIR = os.path.expanduser("~/wedding-recluster")
CLUSTERS_FILE = os.path.join(BASE_DIR, "all_clusters_rebuilt.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUT_DIR = os.path.join(BASE_DIR, "top30_cluster_samples")

TOP_N = 30
SAMPLES_PER_CLUSTER = 5

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
with open(CLUSTERS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = int(row["cluster_id"])
        if cid == -1:
            continue
        rows.append(row)

counts = Counter(int(r["cluster_id"]) for r in rows)
top_clusters = [cid for cid, _ in counts.most_common(TOP_N)]

cluster_to_photos = defaultdict(list)
seen = defaultdict(set)

for row in rows:
    cid = int(row["cluster_id"])
    if cid not in top_clusters:
        continue
    pk = row["photo_key"]
    if pk not in seen[cid]:
        seen[cid].add(pk)
        cluster_to_photos[cid].append(pk)

summary_path = os.path.join(OUT_DIR, "top30_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["cluster_id", "cluster_size", "sample_count", "sample_files"])
    for cid in top_clusters:
        cluster_dir = os.path.join(OUT_DIR, f"cluster_{cid}")
        os.makedirs(cluster_dir, exist_ok=True)

        chosen = cluster_to_photos[cid][:SAMPLES_PER_CLUSTER]
        copied = []

        for i, photo_key in enumerate(chosen, start=1):
            src = os.path.join(IMAGE_DIR, photo_key)
            if not os.path.exists(src):
                continue

            ext = os.path.splitext(photo_key)[1] or ".jpg"
            dst = os.path.join(cluster_dir, f"{i:02d}{ext}")
            shutil.copy2(src, dst)
            copied.append(os.path.basename(dst))

        writer.writerow([cid, counts[cid], len(copied), " | ".join(copied)])

print(f"Done. Output folder: {OUT_DIR}")
print(f"Summary file: {summary_path}")
