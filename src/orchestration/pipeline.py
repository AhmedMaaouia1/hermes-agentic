from typing import List, Tuple
import os

from parsing.file_parser import parse_directory
from agents.analyst import AnalystAgent
from agents.categorizer import CategorizerAgent
from core.types import FileProfile, CategorizationResult


def run_pipeline(folder_path: str) -> Tuple[List[FileProfile], List[CategorizationResult]]:
    """
    Pipeline TEST :
    - Parsing
    - Analyst
    - Categorizer (Agent 2)
    """
    parsed_files = parse_directory(folder_path)

    analyst = AnalystAgent()
    categorizer = CategorizerAgent()

    profiles: List[FileProfile] = []
    categorizations: List[CategorizationResult] = []

    for pf in parsed_files:
        file_path = os.path.join(folder_path, pf.filename)

        # --- Agent Analyst
        profile = analyst.analyze(pf, file_path=file_path)
        profiles.append(profile)

        # --- Agent Catégorisation
        cat = categorizer.categorize(profile)
        categorizations.append(cat)

    return profiles, categorizations
