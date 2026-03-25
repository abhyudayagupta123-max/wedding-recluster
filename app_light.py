from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from search_backend_light import (
    get_photos_for_people,
    list_people,
    get_drive_file_id,
    get_drive_service,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "light API running"}

@app.get("/people")
def people():
    return {"people": list_people()}

@app.get("/search_by_names")
def search_by_names(names: str):
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    photos = get_photos_for_people(name_list)
    return {
        "count": len(photos),
        "photos": photos[:500]
    }

@app.get("/photo")
def photo(name: str):
    fid = get_drive_file_id(name)
    if not fid:
        raise HTTPException(status_code=404, detail="Photo not found")

    service = get_drive_service()

    request = service.files().get_media(fileId=fid)
    data = request.execute()

    return StreamingResponse(
        iter([data]),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'inline; filename="{name}"'
        },
    )

@app.get("/download_photo")
def download_photo(name: str):
    fid = get_drive_file_id(name)
    if not fid:
        raise HTTPException(status_code=404, detail="Photo not found")

    service = get_drive_service()

    request = service.files().get_media(fileId=fid)
    data = request.execute()

    return StreamingResponse(
        iter([data]),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'attachment; filename="{name}"'
        },
    )
