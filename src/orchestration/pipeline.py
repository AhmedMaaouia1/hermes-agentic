from typing import List
import os

from parsing.file_parser import parse_directory
from agents.analyst import AnalystAgent
from core.types import FileProfile


def run_pipeline(folder_path: str) -> List[FileProfile]:
    """
    Pipeline MINIMAL pour tester l'Agent Analyst.
    - Parsing
    - Analyse fichier par fichier
    """
    parsed_files = parse_directory(folder_path)

    analyst = AnalystAgent()
    results: List[FileProfile] = []

    for pf in parsed_files:
        file_path = os.path.join(folder_path, pf.filename)
        profile = analyst.analyze(pf, file_path=file_path)
        results.append(profile)

    return results
