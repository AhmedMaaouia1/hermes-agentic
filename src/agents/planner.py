# src/agents/planner.py

from typing import List, Dict, Optional
import logging

from src.core.types import CategorizationResult, HierarchyProposal


logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Agent 3 — Hierarchy Planner

    Responsabilité :
    - Construire une proposition d'arborescence de dossiers à partir
      des résultats de catégorisation.
    - Raisonnement global, déterministe, sans aucune modification
      des catégories ou sous-catégories.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.6

    def plan(
        self,
        categorizations: List[CategorizationResult],
        existing_tree: Optional[Dict[str, List[str]]] = None
    ) -> HierarchyProposal:
        """
        Génère une proposition hiérarchique de dossiers.

        Args:
            categorizations: Liste des résultats de catégorisation (Agent 2)
            existing_tree: Arborescence existante optionnelle (lecture seule)

        Returns:
            HierarchyProposal strictement conforme à types.py
        """
        folder_structure: Dict[str, List[str]] = {}
        file_to_folder: Dict[str, str] = {}
        warnings: List[str] = []

        logger.info("Starting hierarchy planning for %d files.", len(categorizations))

        for item in categorizations:
            filename = item.filename
            category = item.category
            subcategory = item.subcategory
            confidence = item.confidence

            # Construction du chemin cible
            if subcategory:
                folder_path = f"{category}/{subcategory}"
            else:
                folder_path = category

            # Initialisation de la structure de dossiers
            if folder_path not in folder_structure:
                folder_structure[folder_path] = []

            folder_structure[folder_path].append(filename)
            file_to_folder[filename] = folder_path

            # Warning sur faible confiance
            if confidence < self.LOW_CONFIDENCE_THRESHOLD:
                warning_msg = (
                    f"Low confidence ({confidence:.2f}) for file '{filename}' "
                    f"in category '{category}'"
                )
                warnings.append(warning_msg)
                logger.warning(warning_msg)

        # Détection d'incohérences avec l'arborescence existante (lecture seule)
        if existing_tree:
            for folder, files in folder_structure.items():
                if folder not in existing_tree:
                    warning_msg = f"Proposed new folder not in existing tree: '{folder}'"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                else:
                    for f in files:
                        if f not in existing_tree.get(folder, []):
                            warning_msg = (
                                f"File '{f}' proposed in folder '{folder}' "
                                "differs from existing tree"
                            )
                            warnings.append(warning_msg)
                            logger.warning(warning_msg)

        rationale = (
            "Files are grouped deterministically by their assigned category. "
            "When a subcategory is available, it is used as a second-level folder. "
            "Files without subcategories are placed directly under their category. "
            "No category or subcategory has been modified during planning."
        )

        logger.info("Hierarchy planning completed.")

        return HierarchyProposal(
            folder_structure=folder_structure,
            file_to_folder=file_to_folder,
            rationale=rationale,
            warnings=warnings if warnings else None
        )
