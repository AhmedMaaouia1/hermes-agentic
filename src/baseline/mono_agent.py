from __future__ import annotations
from typing import List, Dict
from src.core.types import ParsedFile, HierarchyProposal, PipelineResult
from src.baseline.rules import FIXED_FOLDERS, choose_folder

def run_baseline(parsed_files: List[ParsedFile]) -> PipelineResult:
    """
    Baseline mono-agent:
    - aucun raisonnement global
    - une passe unique
    - hiérarchie fixe
    """
    folder_structure: Dict[str, List[str]] = {k: [] for k in FIXED_FOLDERS}
    file_to_folder: Dict[str, str] = {}
    warnings: List[str] = []

    for pf in parsed_files:
        folder, w = choose_folder(pf)
        file_to_folder[pf.filename] = folder

        # sécurité : si folder inattendu, on retombe dans Other
        if folder not in folder_structure:
            warnings.append(f"{pf.filename}: folder '{folder}' not in fixed structure -> forced to Other")
            folder = "Other"
            file_to_folder[pf.filename] = folder

        folder_structure[folder].append(pf.filename)
        warnings.extend(w)

    rationale = (
        "Baseline mono-agent (heuristic-only): one-pass local rules based on file extension "
        "and ParsedFile.file_type. Fixed shallow folder structure. No embeddings, no clustering, "
        "no global reasoning."
    )

    proposal = HierarchyProposal(
        folder_structure=folder_structure,
        file_to_folder=file_to_folder,
        rationale=rationale,
        warnings=warnings if warnings else [],
    )

    return PipelineResult(initial_structure=proposal, review=None)
