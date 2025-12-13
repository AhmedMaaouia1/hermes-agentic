import logging
from pathlib import Path


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "hermes.log"

LOG_DIR.mkdir(exist_ok=True)


# ---------- Logging configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


def get_logger(name: str):
    return logging.getLogger(name)
