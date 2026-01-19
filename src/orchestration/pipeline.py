from typing import List, Tuple
import os

from parsing.file_parser import parse_directory
from agents.analyst import AnalystAgent
from agents.categorizer import CategorizerAgent
from agents.planner import PlannerAgent
from agents.reviewer import ReviewerAgent
from core.types import FileProfile, CategorizationResult, HierarchyProposal, ReviewResult, PipelineResult


def run_pipeline(folder_path: str) -> PipelineResult :
    """
    Pipeline HERMES :
    - Parsing des fichiers
    - Analyse (Agent Analyst)
    - Catégorisation sémantique (Agent Categorizer)
    - Planification hiérarchique (Agent Planner)
    - Critique et validation (Agent Reviewer)

    ParsedFiles
     ↓
    AnalystAgent  → FileProfile
        ↓
    CategorizerAgent → CategorizationResult
        ↓
    PlannerAgent → HierarchyProposal
        ↓
    ReviewerAgent → ReviewResult (critique + suggestions)

    """
    parsed_files = parse_directory(folder_path)

    analyst = AnalystAgent()
    categorizer = CategorizerAgent()
    planner = PlannerAgent()
    reviewer = ReviewerAgent()
    profiles: List[FileProfile] = []
    categorizations: List[CategorizationResult] = []
    #hierarchicalplan : List[HierarchyProposal]

    for pf in parsed_files:
        file_path = os.path.join(folder_path, pf.filename)

        # --- Agent Analyst
        profile = analyst.analyze(pf, file_path=file_path)
        profiles.append(profile)

        # --- Agent Catégorisation
        cat = categorizer.categorize(profile)
        categorizations.append(cat)

    # --- Agent Planner
    hierarchy_plan: HierarchyProposal = planner.plan(
        categorizations=categorizations
    )
    # --- Agent Reviewer (lecture seule)
    review_result: ReviewResult = reviewer.review(
        parsed_files=parsed_files,
        profiles=profiles,
        categorizations=categorizations,
        hierarchy=hierarchy_plan,
    )

    reviewer.print_review(review=review_result)
    result_pipeline : PipelineResult = PipelineResult(
        fileprofiles=profiles,
        categorizationRes=categorizations,
        initial_structure=HierarchyProposal,
        review=review_result)
    
    return result_pipeline

