from fastapi import FastAPI, HTTPException
import json
import os

app = FastAPI()

# Load profiles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_PATH = os.path.join(BASE_DIR, "profiles.json")

def load_profiles():
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/profiles")
def get_profiles():
    return load_profiles()

@app.get("/profiles/{user_id}")
def get_profile(user_id: str):
    profiles = load_profiles()
    for p in profiles:
        if p["id"] == user_id:
            return p
    raise HTTPException(status_code=404, detail="Profile not found")
