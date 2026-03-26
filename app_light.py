from PIL import Image
import io
import json
import mimetypes
import os
from functools import lru_cache
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Wedding Gallery Light API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_json(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {filename}")
    with open(path, "r") as f:
        return json.load(f)

cluster_to_photos = load_json("cluster_to_photos.json")
cluster_to_label = load_json("cluster_to_label.json")
photo_to_drive_id = load_json("photo_to_drive_id.json")

person_to_clusters = {}
for cluster_id, person_name in cluster_to_label.items():
    if not person_name:
        continue
    person_name = str(person_name).strip()
    if not person_name or person_name.lower() == "unknown":
        continue
    person_to_clusters.setdefault(person_name, []).append(str(cluster_id))

def list_people():
    return sorted(person_to_clusters.keys(), key=lambda x: x.lower())

def get_photos_for_person(name: str):
    clusters = person_to_clusters.get(name, [])
    photos = set()
    for cluster_id in clusters:
        for photo in cluster_to_photos.get(str(cluster_id), []):
            photos.add(photo)
    return photos

def get_photos_for_people(names, match_mode="all"):
    clean_names = [n.strip() for n in names if n and n.strip()]
    if not clean_names:
        return []

    person_sets = [get_photos_for_person(name) for name in clean_names]

    if match_mode == "any":
        final = set()
        for s in person_sets:
            final |= s
    else:
        final = person_sets[0].copy()
        for s in person_sets[1:]:
            final &= s

    return sorted(final)

def get_all_photos():
    all_photos = set()
    for photos in cluster_to_photos.values():
        for photo in photos:
            all_photos.add(photo)
    return sorted(all_photos)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

@lru_cache(maxsize=1)
def get_drive_service():
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

    creds = service_account.Credentials.from_service_account_info(
        json.loads(raw_json),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)

def build_proxy_path(filename: str):
    return f"/photo?name={quote(filename)}"

@app.get("/")
def root():
    return {"status": "light API running"}

@app.get("/people")
def people():
    return {"people": list_people(), "count": len(list_people())}

@app.get("/search_by_names")
def search_by_names(
    names: str = Query("", description="Comma-separated names"),
    mode: str = Query("all", description="all or any"),
):
    selected_names = [n.strip() for n in names.split(",") if n.strip()]

    if not selected_names:
        photos = get_all_photos()
        proxy_photos = [build_proxy_path(photo) for photo in photos]
        return {
            "count": len(proxy_photos),
            "photos": proxy_photos,
            "names": [],
            "mode": "all",
        }

    photos = get_photos_for_people(selected_names, match_mode=mode.lower())
    proxy_photos = [build_proxy_path(photo) for photo in photos]

    return {
        "count": len(proxy_photos),
        "photos": proxy_photos,
        "names": selected_names,
        "mode": mode.lower(),
    }

@app.get("/photo")
def get_photo(name: str):
    if name not in photo_to_drive_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_id = photo_to_drive_id[name]
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"

    headers = {
        "Cache-Control": "public, max-age=31536000, immutable"
    }
    return StreamingResponse(file_stream, media_type=media_type, headers=headers)

@app.get("/download_photo")
def download_photo(name: str):
    if name not in photo_to_drive_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_id = photo_to_drive_id[name]
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"

    headers = {
        "Content-Disposition": f'attachment; filename="{name}"',
        "Cache-Control": "public, max-age=31536000, immutable"
    }
    return StreamingResponse(file_stream, media_type=media_type, headers=headers)

@app.get("/photo_thumb")
def photo_thumb(name: str):
    if name not in photo_to_drive_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_id = photo_to_drive_id[name]
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)

    # Create thumbnail
    img = Image.open(file_stream)
    img.thumbnail((400, 400))  # 👈 key line

    thumb_stream = io.BytesIO()
    img.save(thumb_stream, format="JPEG", quality=70)
    thumb_stream.seek(0)

    headers = {
        "Cache-Control": "public, max-age=31536000, immutable"
    }

    return StreamingResponse(thumb_stream, media_type="image/jpeg", headers=headers)



@app.get("/photo_display")
def photo_display(name: str, max_dim: int = 1200, quality: int = 65):
    if name not in photo_to_drive_id:
        raise HTTPException(status_code=404, detail="Photo not found")

    service = get_drive_service()
    file_id = photo_to_drive_id[name]
    request = service.files().get_media(fileId=file_id)

    original_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(original_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    original_stream.seek(0)

    try:
        with Image.open(original_stream) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            img.thumbnail((max_dim, max_dim))

            out = io.BytesIO()
            img.save(out, format="JPEG", quality=max(35, min(quality, 85)), progressive=True)
            out.seek(0)

            headers = {
                "Cache-Control": "public, max-age=604800"
            }
            return StreamingResponse(out, media_type="image/jpeg", headers=headers)

    except Exception:
        original_stream.seek(0)
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        headers = {
            "Cache-Control": "public, max-age=604800"
        }
        return StreamingResponse(original_stream, media_type=media_type, headers=headers)

