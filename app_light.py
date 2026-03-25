from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search_backend_light import get_photos_for_people, list_people

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
