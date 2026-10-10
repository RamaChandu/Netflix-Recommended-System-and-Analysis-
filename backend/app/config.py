
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv.zip"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:5175", "http://127.0.0.1:5175",
]
