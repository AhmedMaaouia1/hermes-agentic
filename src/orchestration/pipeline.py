from typing import List, Tuple
import os

from parsing.file_parser import parse_directory
from agents.analyst import AnalystAgent
from agents.categorizer import CategorizerAgent
from agents.planner import PlannerAgent
from core.types import FileProfile, CategorizationResult, HierarchyProposal


def run_pipeline(folder_path: str) -> Tuple[List[FileProfile], List[CategorizationResult], HierarchyProposal]:
    """
    Pipeline TEST :
    - Parsing
    - Analyst
    - Categorizer (Agent 2)
    - Planification hiérarchique (Agent Planner : Agent 3)
    """
    parsed_files = parse_directory(folder_path)

    analyst = AnalystAgent()
    categorizer = CategorizerAgent()
    planner = PlannerAgent()

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
    hierarchy_plan = planner.plan(categorizations=categorizations)


    return profiles, categorizations, hierarchy_plan
