import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "saved_models"
REPORT_DIR = PROJECT_ROOT / "reports"
STATIC_GEN_DIR = PROJECT_ROOT / "static" / "generated"

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, STATIC_GEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
