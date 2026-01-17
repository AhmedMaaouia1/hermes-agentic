from __future__ import annotations
from typing import List, Tuple
from src.core.types import ParsedFile

# Hiérarchie FIXE : clé = dossier, valeur = liste de fichiers (remplie ailleurs)
FIXED_FOLDERS = [
    "Documents/PDF",
    "Documents/Office",
    "Documents/Text",
    "Images",
    "Archives",
    "Code",
    "Executables",
    "Other",
]

OFFICE_EXT = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".odp", ".ods"}
TEXT_EXT = {".txt", ".md", ".rtf", ".csv"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".php", ".html", ".css", ".json", ".yaml", ".yml", ".ipynb"}
ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz"}

def choose_folder(pf: ParsedFile) -> Tuple[str, List[str]]:
    """
    Baseline : décision STRICTEMENT LOCALE.
    Retourne (folder_path, warnings_for_this_file)
    """
    warnings: List[str] = []

    ext = pf.extension.lower().strip() if pf.extension else ""
    tokens = pf.tokens or []

    if pf.has_copy:
        warnings.append(f"{pf.filename}: detected copy marker in name")

    if not tokens:
        warnings.append(f"{pf.filename}: no tokens extracted from filename")

    # 1) Cas directs par extension (très naïf)
    if ext == ".pdf":
        return "Documents/PDF", warnings

    if ext in OFFICE_EXT:
        return "Documents/Office", warnings

    if ext in TEXT_EXT:
        return "Documents/Text", warnings

    if ext in ARCHIVE_EXT:
        return "Archives", warnings

    if ext in CODE_EXT:
        return "Code", warnings

    # 2) Cas par type générique (si ton parser le remplit correctement)
    ft = (pf.file_type or "").lower()
    if ft == "image":
        return "Images", warnings
    if ft == "executable":
        return "Executables", warnings
    if ft == "document":
        # fallback document naïf
        return "Documents/Text", warnings

    # 3) Inconnu
    if not ext or ext == ".":
        warnings.append(f"{pf.filename}: missing/invalid extension")
    else:
        warnings.append(f"{pf.filename}: unknown extension {ext}")

    return "Other", warnings