import os
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHOTOS_FILE = os.path.join(BASE_DIR, "cluster_to_photos.json")
LABEL_FILE = os.path.join(BASE_DIR, "cluster_to_label.json")
DRIVE_FILE = os.path.join(BASE_DIR, "photo_to_drive_id.json")

with open(PHOTOS_FILE, "r", encoding="utf-8") as f:
    cluster_to_photos = {int(k): set(v) for k, v in json.load(f).items()}

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    cluster_to_label = {int(k): v for k, v in json.load(f).items()}

with open(DRIVE_FILE, "r", encoding="utf-8") as f:
    photo_to_drive = json.load(f)

name_to_clusters = defaultdict(list)
for cid, name in cluster_to_label.items():
    name_to_clusters[name].append(cid)

def list_people():
    return sorted(name_to_clusters.keys())

def get_drive_link(filename):
    fid = photo_to_drive.get(filename)
    if not fid:
        return None
    return f"https://drive.google.com/uc?id={fid}"

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
    photos = sorted(result)

    return [get_drive_link(p) for p in photos if get_drive_link(p)]
