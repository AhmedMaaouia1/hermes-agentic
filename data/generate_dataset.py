import os
import random
import string
from datetime import datetime, timedelta

BASE_DIR = "data/downloads_raw"
os.makedirs(BASE_DIR, exist_ok=True)

random.seed(42)

# ---------- Helpers ----------
def random_string(n=5):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def touch_file(path):
    with open(path, "w") as f:
        f.write("")

# ---------- File templates ----------
file_templates = [
    ("facture_electricite", "pdf"),
    ("facture_internet", "pdf"),
    ("impots_2023", "pdf"),
    ("attestation_scolarite", "pdf"),
    ("carte_identite", "pdf"),
    ("CV_stage_2025", "pdf"),
    ("cours_nlp_ch", "pdf"),
    ("cours_machine_learning", "pdf"),
    ("TD_IA", "docx"),
    ("projet_ia_rapport", "pdf"),
    ("projet_code", "zip"),
    ("dataset_resultats", "xlsx"),
    ("notes_reunion", "txt"),
    ("script_analysis", "py"),
    ("installer_app", "exe"),
    ("IMG", "jpeg"),
    ("screenshot_error", "png")
]

# ---------- Generate clean files ----------
files = []

for i in range(120):
    name, ext = random.choice(file_templates)
    filename = f"{name}_{i}.{ext}"
    files.append(filename)
    touch_file(os.path.join(BASE_DIR, filename))

# ---------- Add noisy filenames ----------
noise_patterns = [
    "scan",
    "doc_final",
    "final_v2",
    "new",
    "copy",
    "temp"
]

for i in range(20):
    noise = random.choice(noise_patterns)
    ext = random.choice(["pdf", "png", "jpeg", "txt"])
    filename = f"{noise}_{random_string()}.{ext}"
    files.append(filename)
    touch_file(os.path.join(BASE_DIR, filename))

# ---------- Add ambiguous files ----------
ambiguous_files = [
    "scan123.pdf",
    "document.pdf",
    "file_final.pdf",
    "notes.pdf",
    "image.jpeg",
    "archive.zip",
    "data.xlsx",
    "script.py",
    "installer.exe",
    "unknown_file.txt"
]

for filename in ambiguous_files:
    files.append(filename)
    touch_file(os.path.join(BASE_DIR, filename))

# ---------- Add duplicates ----------
for i in range(5):
    original = random.choice(files)
    name, ext = original.split(".")
    dup = f"{name}_copy.{ext}"
    touch_file(os.path.join(BASE_DIR, dup))

print(f"Dataset generated with {len(os.listdir(BASE_DIR))} files.")
