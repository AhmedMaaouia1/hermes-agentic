from pydantic import BaseModel
from typing import Optional, List, Dict


# ---------- Entrée brute ----------
class FileInfo(BaseModel):
    filename: str
    extension: str
    size_kb: Optional[int]
    created_at: Optional[str]
    modified_at: Optional[str]


# ---------- Après analyse ----------
class FileProfile(BaseModel):
    filename: str
    file_type: str              # document, image, code, archive, other
    topic: str                  # ex: facture, cours, projet IA
    keywords: List[str]


# ---------- Catégorisation ----------
class CategorizationResult(BaseModel):
    filename: str
    category: str               # Administratif, Cours, Projets, Images, etc.
    subcategory: Optional[str]
    confidence: float           # entre 0 et 1
    rationale: str              # justification textuelle


# ---------- Hiérarchie ----------
class HierarchyProposal(BaseModel):
    folder_structure: Dict[str, List[str]]
    file_to_folder: Dict[str, str]
    rationale: str


# ---------- Critique / Validation ----------
class ReviewResult(BaseModel):
    issues: List[str]
    suggestions: List[str]
    revised_structure: Optional[HierarchyProposal]


# ---------- Résultat final ----------
class PipelineResult(BaseModel):
    initial_structure: HierarchyProposal
    review: Optional[ReviewResult]
