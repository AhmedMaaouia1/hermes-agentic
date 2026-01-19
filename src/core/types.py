"""
types.py

Ce fichier définit TOUS les types de données échangés entre les composants
du projet HERMES Agentic.

Règle d'or :
- Chaque agent communique uniquement via ces structures.
- Aucun agent ne doit "deviner" ou modifier le format.
- Toute comparaison mono-agent vs multi-agents repose sur ces types.

Pipeline conceptuel :
Fichier brut
→ Parsing neutre (ParsedFile)
→ Analyse (FileProfile)
→ Catégorisation (CategorizationResult)
→ Hiérarchie (HierarchyProposal)
→ (Optionnel) Critique / Validation (ReviewResult)
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ======================================================
# 1. SORTIE DU PARSING (NEUTRE, DÉTERMINISTE)
# ======================================================
class ParsedFile(BaseModel):
    """
    Représentation neutre d’un fichier après parsing.
    Produite par le module de parsing commun.
    
    Aucune décision sémantique ici.
    Aucune IA.
    Aucune logique métier.
    """

    filename: str                       # nom du fichier (ex: facture_2023.pdf)
    extension: str                      # extension (ex: .pdf)
    normalized_name: str                # nom normalisé (lowercase, sans symboles)
    tokens: List[str]                   # tokens simples extraits du nom
    file_type: str                      # document, image, code, archive, executable, other
    # signaux temporels (passifs)
    created_at: Optional[str]           # ISO format
    modified_at: Optional[str]          # ISO format
    has_year: Optional[int]             # année détectée (ex: 2023) ou None
    has_copy: bool                      # présence de "copy", "copie", etc.


# ======================================================
# 2. SORTIE DE L’AGENT ANALYST (ANALYSE SÉMANTIQUE LOCALE)
# ======================================================
class FileProfile(BaseModel):
    """
    Profil sémantique léger d’un fichier.
    Produit par l’Agent Analyst (Agent 1).

    Ici, on introduit les premières hypothèses sémantiques,
    mais sans décider de la taxonomie finale.
    """

    filename: str
    file_type: str                      # repris du parsing
    topic: Optional[str]                # ex: facture, cours, cv, projet
    keywords: List[str]                 # mots-clés extraits ou inférés
    signals: Dict[str, Any]             # signaux faibles (ex: {"source": "filename"})
    summary: Optional[str]              # Résumé factuel court, descriptif, non décisionnel


# ======================================================
# 3. SORTIE DE L’AGENT CATÉGORISATION (TAXONOMIE)
# ======================================================
class CategorizationResult(BaseModel):
    """
    Résultat de classification sémantique dans la taxonomie officielle.
    Produit par l’Agent Catégorisation (Agent 2).
    """

    filename: str
    category: str                       # catégorie officielle (ex: Administratif)
    subcategory: Optional[str]          # sous-catégorie officielle (ex: Factures)
    confidence: float                   # score de confiance [0, 1]
    rationale: str                      # justification explicable
    decision_source: str                # rule / embedding / hybrid


# ======================================================
# 4. SORTIE DE L’AGENT HIÉRARCHIE (STRUCTURATION GLOBALE)
# ======================================================
class HierarchyProposal(BaseModel):
    """
    Proposition d’organisation globale des fichiers.
    Produite par l’Agent Hiérarchie (Agent 3).

    L’agent raisonne sur l’ensemble des fichiers,
    jamais fichier par fichier.
    """

    folder_structure: Dict[str, List[str]]
    # ex:
    # {
    #   "Administratif/Factures": ["facture_1.pdf", "facture_2.pdf"],
    #   "Cours/NLP": ["cours_nlp_ch_12.pdf"]
    # }

    file_to_folder: Dict[str, str]
    # ex:
    # {
    #   "facture_1.pdf": "Administratif/Factures",
    #   "cours_nlp_ch_12.pdf": "Cours/NLP"
    # }

    rationale: str                      # principes globaux utilisés
    warnings: Optional[List[str]] = []  # ambiguïtés, faibles confiances, collisions


# ======================================================
# 5. SORTIE DE L’AGENT REVIEWER / CRITIQUE (OPTIONNEL)
# ======================================================
class ReviewIssue(BaseModel):
    """
    Problème identifié par l’Agent Reviewer.
    """

    issue_type: str                     # ex: misclassification, ambiguity, inconsistency
    severity: str                       # low / medium / high
    description: str                    # explication claire
    affected_files: List[str]


class ReviewSuggestion(BaseModel):
    """
    Suggestion d’amélioration proposée par l’Agent Reviewer.
    """

    action: str                         # move_file / merge_folder / rename_folder
    target: str                         # fichier ou dossier concerné
    suggestion: str                    # description textuelle


class ReviewResult(BaseModel):
    """
    Résultat de la phase de critique / validation.
    Produit par l’Agent Reviewer (Agent 4).

    L’agent critique NE MODIFIE PAS directement la hiérarchie :
    il propose, le système décide.
    """

    issues: List[ReviewIssue]
    suggestions: List[ReviewSuggestion]
    revised_structure: Optional[HierarchyProposal] = None


# ======================================================
# 6. SORTIE FINALE DU PIPELINE
# ======================================================
class PipelineResult(BaseModel):
    """
    Résultat final du système HERMES Agentic.
    Utilisé pour :
    - affichage
    - export JSON
    - évaluation
    - comparaison mono-agent vs multi-agents
    """
    fileprofiles: FileProfile
    categorizationRes : CategorizationResult
    initial_structure: HierarchyProposal
    review: Optional[ReviewResult] = None

