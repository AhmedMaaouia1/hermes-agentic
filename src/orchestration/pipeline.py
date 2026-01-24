"""
pipeline.py

Pipeline principal de HERMES Agentic.

Responsabilité UNIQUE :
- Orchestrer séquentiellement les agents
- Garantir l'ordre, la cohérence et la traçabilité
- NE PAS contenir de logique métier ou sémantique

Règle d'or :
- Chaque agent écrit ses propres résultats sur disque
  immédiatement après son exécution.
- Le pipeline ne fait que coordonner et agréger.
"""

from typing import List
import os
import json
from pathlib import Path

from parsing.file_parser import parse_directory
from agents.analyst import AnalystAgent
from agents.categorizer import CategorizerAgent
from agents.planner import PlannerAgent
from agents.reviewer import ReviewerAgent

from core.types import (
    ParsedFile,
    FileProfile,
    CategorizationResult,
    HierarchyProposal,
    ReviewResult,
    PipelineResult,
)


# ======================================================
# Configuration globale des logs
# ======================================================

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def _safe_write_json(path: Path, data) -> None:
    """
    Écriture JSON robuste et lisible.

    Bonne pratique :
    - centraliser l'écriture
    - éviter la duplication de code
    - garantir UTF-8 + indent stable
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================================================
# Pipeline principal
# ======================================================

def run_pipeline(folder_path: str) -> PipelineResult:
    """
    Exécute le pipeline HERMES Agentic de bout en bout.

    Étapes :
    1) Parsing neutre (ParsedFile)
    2) Analyse locale (Agent Analyst → FileProfile)
    3) Catégorisation taxonomique (Agent Categorizer → CategorizationResult)
    4) Planification globale (Agent Planner → HierarchyProposal)
    5) Revue critique (Agent Reviewer → ReviewResult)

    Important :
    - Chaque agent écrit ses sorties immédiatement
    - Le pipeline retourne un PipelineResult pour usage CLI / tests
    """

    # --------------------------------------------------
    # 0) Vérifications de base
    # --------------------------------------------------
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input path does not exist or is not a directory: {folder_path}")

    # --------------------------------------------------
    # 1) Parsing (neutre, déterministe)
    # --------------------------------------------------
    parsed_files: List[ParsedFile] = parse_directory(folder_path)

    if not parsed_files:
        raise RuntimeError("No files found after parsing. Pipeline aborted.")

    _safe_write_json(
        LOGS_DIR / "parser" / "parsed_files.json",
        [p.model_dump() for p in parsed_files],
    )

    # --------------------------------------------------
    # 2) Initialisation des agents
    # --------------------------------------------------
    analyst = AnalystAgent()
    categorizer = CategorizerAgent()
    planner = PlannerAgent()
    reviewer = ReviewerAgent(
        enable_llm=True,          # LLM optionnel et non décisionnel
        llm_model="mistral:7b",   # modèle local via Ollama
    )

    profiles: List[FileProfile] = []
    categorizations: List[CategorizationResult] = []

    # --------------------------------------------------
    # 3) Analyse + Catégorisation (fichier par fichier)
    # --------------------------------------------------
    for pf in parsed_files:
        file_path = os.path.join(folder_path, pf.filename)

        # --- Agent Analyst
        profile = analyst.analyze(pf, file_path=file_path)
        profiles.append(profile)

        # --- Agent Categorizer
        cat = categorizer.categorize(profile)
        categorizations.append(cat)

    # --- Persist Analyst output
    _safe_write_json(
        LOGS_DIR / "analyst" / "file_profiles.json",
        [p.model_dump() for p in profiles],
    )

    # --- Persist Categorizer output
    _safe_write_json(
        LOGS_DIR / "categorizer" / "categorizations.json",
        [c.model_dump() for c in categorizations],
    )

    # --------------------------------------------------
    # 4) Planification hiérarchique (raisonnement global)
    # --------------------------------------------------
    hierarchy_plan: HierarchyProposal = planner.plan(
        categorizations=categorizations
    )

    _safe_write_json(
        LOGS_DIR / "planner" / "hierarchy.json",
        hierarchy_plan.model_dump(),
    )

    # --------------------------------------------------
    # 5) Revue critique (lecture seule)
    # --------------------------------------------------
    review_result: ReviewResult = reviewer.review(
        parsed_files=parsed_files,
        profiles=profiles,
        categorizations=categorizations,
        hierarchy=hierarchy_plan,
    )

    _safe_write_json(
        LOGS_DIR / "reviewer" / "review.json",
        review_result.model_dump(),
    )

    # Optionnel : affichage console pour debug / démonstration
    from agents.reviewer import print_review
    print_review(review_result)

    # --------------------------------------------------
    # 6) Agrégation finale (sans logique métier)
    # --------------------------------------------------
    return PipelineResult(
        parsed_files=parsed_files,
        file_profiles=profiles,
        categorizations=categorizations,
        hierarchy=hierarchy_plan,
        review=review_result,
    )
