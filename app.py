from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from search_backend import get_photos_for_people, search_by_uploaded_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "API running"}


@app.get("/search_by_names")
def search_by_names(names: str):
    name_list = [n.strip() for n in names.split(",")]
    photos = get_photos_for_people(name_list)
    return {
        "count": len(photos),
        "photos": photos[:200]
    }


@app.post("/search_by_upload")
def search_by_upload(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = search_by_uploaded_face(file_path)

    return result
