"""
file_parser.py

Module de parsing commun et neutre pour le projet HERMES Agentic.

Responsabilité :
- Extraire des informations syntaxiques simples depuis les fichiers
- Aucune IA
- Aucune décision sémantique
- Utilisé par la baseline mono-agent et l’Agent Analyst
"""

import os
import re
from datetime import datetime
from typing import List, Optional

from src.core.types import ParsedFile


# ------------------------------------------------------
# Extensions connues (classification grossière)
# ------------------------------------------------------
DOCUMENT_EXT = {".pdf", ".docx", ".txt", ".xlsx", ".pptx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
CODE_EXT = {".py", ".js", ".java"}
ARCHIVE_EXT = {".zip", ".rar", ".tar"}
EXECUTABLE_EXT = {".exe", ".msi"}


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def detect_file_type(extension: str) -> str:
    """Détermine un type de fichier grossier à partir de l’extension."""
    if extension in DOCUMENT_EXT:
        return "document"
    if extension in IMAGE_EXT:
        return "image"
    if extension in CODE_EXT:
        return "code"
    if extension in ARCHIVE_EXT:
        return "archive"
    if extension in EXECUTABLE_EXT:
        return "executable"
    return "other"


def normalize_filename(filename: str) -> str:
    """Normalise un nom de fichier (lowercase, séparateurs unifiés)."""
    name = filename.lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def extract_year(tokens: List[str]) -> Optional[int]:
    """Extrait une année plausible depuis une liste de tokens."""
    current_year = datetime.now().year + 1
    for tok in tokens:
        if tok.isdigit() and len(tok) == 4:
            year = int(tok)
            if 1990 <= year <= current_year:
                return year
    return None


# ------------------------------------------------------
# Parsing principal
# ------------------------------------------------------
def parse_file(file_path: str) -> ParsedFile:
    """
    Parse un fichier unique et retourne un ParsedFile.
    """
    filename = os.path.basename(file_path)
    name, extension = os.path.splitext(filename)
    extension = extension.lower()

    normalized_name = normalize_filename(name)
    tokens = normalized_name.split(" ")

    has_copy = any(tok in {"copy", "copie", "duplicate"} for tok in tokens)
    year = extract_year(tokens)

    return ParsedFile(
        filename=filename,
        extension=extension,
        normalized_name=normalized_name,
        tokens=tokens,
        file_type=detect_file_type(extension),
        has_year=year,
        has_copy=has_copy,
    )


def parse_directory(directory_path: str) -> List[ParsedFile]:
    """
    Parse tous les fichiers d’un dossier (niveau 1 uniquement).
    """
    parsed_files = []

    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            parsed_files.append(parse_file(full_path))

    return parsed_files
