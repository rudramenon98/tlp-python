from pathlib import Path

CURR_DIR = Path(__file__).parent
DATA_DIR = CURR_DIR.parents[3] / "data"
OUTPUT_DIR = CURR_DIR.parents[3] / "output"
DOCUMENTS_DIR = DATA_DIR / "documents"
MODEL_DIR = OUTPUT_DIR / "models"
