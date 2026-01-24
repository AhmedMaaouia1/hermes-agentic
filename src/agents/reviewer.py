"""
reviewer.py

HERMES Agentic — Agent Reviewer / Critique (Agent 4)

Responsabilité (STRICTE) :
- Prendre les sorties des agents précédents (ParsedFile, FileProfile, CategorizationResult, HierarchyProposal)
- Détecter des problèmes GLOBAUX (incohérences, ambiguïtés, faible confiance, collisions)
- Proposer des suggestions ACTIONNABLES (sans appliquer de modifications)

Règles académiques :
- Ne modifie PAS la taxonomie (pas de recatégorisation)
- Ne modifie PAS directement la hiérarchie (revised_structure=None par défaut)
- Garde une logique défendable et traçable
- LLM optionnel : uniquement pour reformuler des suggestions (pas pour décider)

Sortie :
- ReviewResult conforme à core/types.py
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# Imports robustes selon votre exécution
try:
    from core.types import (
        ParsedFile,
        FileProfile,
        CategorizationResult,
        HierarchyProposal,
        ReviewIssue,
        ReviewSuggestion,
        ReviewResult,
    )
except ModuleNotFoundError:
    from src.core.types import (
        ParsedFile,
        FileProfile,
        CategorizationResult,
        HierarchyProposal,
        ReviewIssue,
        ReviewSuggestion,
        ReviewResult,
    )

logger = logging.getLogger("hermes.reviewer")


# ======================================================
# Utils
# ======================================================
def _safe_lower(s: Optional[str]) -> str:
    return (s or "").lower()


def _folder_depth(path: str) -> int:
    # "A/B/C" -> depth 3
    return len([p for p in path.split("/") if p.strip()])


def _normalize_token(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9àâäçéèêëîïôöùûüÿñæœ ]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_base_name(filename: str) -> str:
    # "facture_2023 (copy).pdf" -> "facture_2023 (copy)"
    base = os.path.splitext(filename)[0]
    return base.strip()


def _looks_like_duplicate_name(parsed: ParsedFile) -> bool:
    # Use parsing signal + common patterns
    name = _safe_lower(parsed.filename)
    patterns = [
        r"\bcopy\b", r"\bcopie\b", r"\bduplicate\b", r"\bdup\b",
        r"\(\d+\)$",               # "file (1)"
        r"\b\d+_copy\b",
    ]
    if parsed.has_copy:
        return True
    return any(re.search(p, name) for p in patterns)

# ======================================================
# Reviewer Agent
# ======================================================
class ReviewerAgent:
    """
    Agent 4 — Critique / Validation (global)

    Input:
    - parsed_files: List[ParsedFile]
    - profiles: List[FileProfile]
    - categorizations: List[CategorizationResult]
    - hierarchy: HierarchyProposal

    Output:
    - ReviewResult
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.55,
        heterogeneous_folder_threshold: int = 2,
        max_folder_depth_allowed: int = 3,
        enable_llm: bool = True,
        llm_model: str = "mistral:7b",
    ):
        """
        Args:
            low_confidence_threshold: en dessous -> issue "low_confidence"
            heterogeneous_folder_threshold: nb de catégories distinctes tolérées par dossier avant warning
            max_folder_depth_allowed: contrôle de complexité (2-3 niveaux max conseillé)
        """
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.heterogeneous_folder_threshold = int(heterogeneous_folder_threshold)
        self.max_folder_depth_allowed = int(max_folder_depth_allowed)
        
        self.enable_llm = enable_llm
        self.llm_model = llm_model
    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def review(
        self,
        parsed_files: List[ParsedFile],
        profiles: List[FileProfile],
        categorizations: List[CategorizationResult],
        hierarchy: HierarchyProposal,
    ) -> ReviewResult:
        logger.info("ReviewerAgent: starting review")

        issues: List[ReviewIssue] = []
        suggestions: List[ReviewSuggestion] = []

        parsed_by_name: Dict[str, ParsedFile] = {p.filename: p for p in parsed_files}
        profile_by_name: Dict[str, FileProfile] = {p.filename: p for p in profiles}
        cat_by_name: Dict[str, CategorizationResult] = {c.filename: c for c in categorizations}

        # 1) Sanity checks (coverage)
        issues.extend(self._check_coverage(parsed_by_name, profile_by_name, cat_by_name, hierarchy))

        # 2) Low confidence categorizations
        issues.extend(self._detect_low_confidence(cat_by_name))

        # 3) Unknown/Divers concentration warnings
        issues.extend(self._detect_divers_overuse(hierarchy, cat_by_name))

        # 4) Folder heterogeneity (many categories in same folder)
        issues.extend(self._detect_heterogeneous_folders(hierarchy, cat_by_name))

        # 5) Duplicates & naming collisions
        issues.extend(self._detect_duplicates(parsed_by_name, hierarchy))

        # 6) Hierarchy complexity checks
        issues.extend(self._check_hierarchy_complexity(hierarchy))

        # 7) Suggestions (actionable, no direct edits)
        suggestions.extend(self._suggest_merge_small_folders(hierarchy))
        suggestions.extend(self._suggest_review_folder_for_low_confidence(hierarchy, cat_by_name))
        suggestions.extend(self._suggest_rename_suspicious_folders(hierarchy))
        
        # 8) LLM reformulation (OPTIONNELLE, non décisionnelle)
        if self.enable_llm and suggestions:
            suggestions = self._llm_rewrite_suggestions(
                suggestions=suggestions,
                hierarchy=hierarchy,
            )

        logger.info("ReviewerAgent: review completed issues=%d suggestions=%d", len(issues), len(suggestions))

        return ReviewResult(
            issues=issues,
            suggestions=suggestions,
            revised_structure=None,  # IMPORTANT: no automatic modification
        )

    # --------------------------------------------------
    # Checks
    # --------------------------------------------------
    def _check_coverage(
        self,
        parsed_by_name: Dict[str, ParsedFile],
        profile_by_name: Dict[str, FileProfile],
        cat_by_name: Dict[str, CategorizationResult],
        hierarchy: HierarchyProposal,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        parsed_set = set(parsed_by_name.keys())
        prof_set = set(profile_by_name.keys())
        cat_set = set(cat_by_name.keys())
        hier_set = set(hierarchy.file_to_folder.keys())

        # Missing links
        missing_profiles = sorted(list(parsed_set - prof_set))
        if missing_profiles:
            issues.append(
                ReviewIssue(
                    issue_type="missing_profiles",
                    severity="high",
                    description=f"{len(missing_profiles)} fichiers parsés n'ont pas de FileProfile (Agent Analyst).",
                    affected_files=missing_profiles,
                )
            )

        missing_cats = sorted(list(prof_set - cat_set))
        if missing_cats:
            issues.append(
                ReviewIssue(
                    issue_type="missing_categorizations",
                    severity="high",
                    description=f"{len(missing_cats)} FileProfiles n'ont pas de CategorizationResult (Agent 2).",
                    affected_files=missing_cats,
                )
            )

        missing_in_hierarchy = sorted(list(cat_set - hier_set))
        if missing_in_hierarchy:
            issues.append(
                ReviewIssue(
                    issue_type="missing_in_hierarchy",
                    severity="high",
                    description=f"{len(missing_in_hierarchy)} fichiers catégorisés ne sont pas routés dans la hiérarchie.",
                    affected_files=missing_in_hierarchy,
                )
            )

        # Orphan routing (file in hierarchy but no categorization)
        orphan_in_hierarchy = sorted(list(hier_set - cat_set))
        if orphan_in_hierarchy:
            issues.append(
                ReviewIssue(
                    issue_type="orphan_in_hierarchy",
                    severity="medium",
                    description=f"{len(orphan_in_hierarchy)} fichiers routés n'ont pas de catégorisation (vérifier pipeline).",
                    affected_files=orphan_in_hierarchy,
                )
            )

        return issues

    def _detect_low_confidence(self, cat_by_name: Dict[str, CategorizationResult]) -> List[ReviewIssue]:
        low = [c.filename for c in cat_by_name.values() if float(c.confidence) < self.low_confidence_threshold]
        if not low:
            return []
        return [
            ReviewIssue(
                issue_type="low_confidence_categorization",
                severity="medium",
                description=(
                    f"{len(low)} fichiers ont une confiance < {self.low_confidence_threshold:.2f}. "
                    "Ils sont candidats pour une vérification manuelle."
                ),
                affected_files=sorted(low),
            )
        ]

    def _detect_divers_overuse(
        self,
        hierarchy: HierarchyProposal,
        cat_by_name: Dict[str, CategorizationResult],
    ) -> List[ReviewIssue]:
        divers_files = []
        for fn in hierarchy.file_to_folder.keys():
            c = cat_by_name.get(fn)
            if not c:
                continue
            if _normalize_token(c.category) in {"divers", "autre", "other", "misc"}:
                divers_files.append(fn)

        if not divers_files:
            return []

        severity = "low"
        if len(divers_files) >= 0.3 * max(1, len(cat_by_name)):
            severity = "medium"

        return [
            ReviewIssue(
                issue_type="divers_overuse",
                severity=severity,
                description=(
                    f"{len(divers_files)} fichiers sont classés en 'Divers'. "
                    "Cela peut indiquer des règles insuffisantes ou un profil trop pauvre."
                ),
                affected_files=sorted(divers_files),
            )
        ]

    def _detect_heterogeneous_folders(
        self,
        hierarchy: HierarchyProposal,
        cat_by_name: Dict[str, CategorizationResult],
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for folder, files in hierarchy.folder_structure.items():
            cats = []
            for fn in files:
                c = cat_by_name.get(fn)
                if c:
                    cats.append(c.category)
            uniq = sorted(set(cats))

            if len(uniq) > self.heterogeneous_folder_threshold:
                issues.append(
                    ReviewIssue(
                        issue_type="heterogeneous_folder",
                        severity="medium",
                        description=(
                            f"Le dossier '{folder}' contient {len(uniq)} catégories différentes: {uniq}. "
                            "Cela peut réduire la cohérence de l'arborescence."
                        ),
                        affected_files=sorted(files),
                    )
                )

        return issues

    def _detect_duplicates(
        self,
        parsed_by_name: Dict[str, ParsedFile],
        hierarchy: HierarchyProposal,
    ) -> List[ReviewIssue]:
        """
        Détecte des duplicats probables via patterns + tokens + has_copy.
        But: signaler, pas supprimer.
        """
        issues: List[ReviewIssue] = []

        candidates = []
        for fn in hierarchy.file_to_folder.keys():
            p = parsed_by_name.get(fn)
            if not p:
                continue
            if _looks_like_duplicate_name(p):
                candidates.append(fn)

        if candidates:
            issues.append(
                ReviewIssue(
                    issue_type="possible_duplicates",
                    severity="low",
                    description="Certains fichiers semblent être des doublons (copy/copie/(1), etc.).",
                    affected_files=sorted(candidates),
                )
            )

        # collisions base-name (same base name, different extensions)
        base_map: Dict[str, List[str]] = defaultdict(list)
        for fn in hierarchy.file_to_folder.keys():
            base_map[_normalize_token(_extract_base_name(fn))].append(fn)

        collisions = [v for v in base_map.values() if len(v) >= 2]
        # keep only non-trivial (different ext or copy pattern)
        collision_files = []
        for group in collisions:
            exts = set(os.path.splitext(f)[1].lower() for f in group)
            if len(exts) >= 2 or any("copy" in f.lower() or "copie" in f.lower() for f in group):
                collision_files.extend(group)

        if collision_files:
            issues.append(
                ReviewIssue(
                    issue_type="naming_collision",
                    severity="low",
                    description="Collisions de noms détectées (même nom de base avec extensions différentes ou copies).",
                    affected_files=sorted(set(collision_files)),
                )
            )

        return issues

    def _check_hierarchy_complexity(self, hierarchy: HierarchyProposal) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []
        depths = [_folder_depth(f) for f in hierarchy.folder_structure.keys()]
        if not depths:
            return issues

        max_depth = max(depths)
        if max_depth > self.max_folder_depth_allowed:
            issues.append(
                ReviewIssue(
                    issue_type="hierarchy_too_deep",
                    severity="medium",
                    description=(
                        f"La hiérarchie contient des chemins profonds (max_depth={max_depth}). "
                        f"Recommandé: <= {self.max_folder_depth_allowed}."
                    ),
                    affected_files=[],
                )
            )
        return issues

    # --------------------------------------------------
    # Suggestions
    # --------------------------------------------------
    def _suggest_merge_small_folders(self, hierarchy: HierarchyProposal) -> List[ReviewSuggestion]:
        """
        Si un dossier a 1 seul fichier, proposer un merge "avec le dossier parent".
        (Suggestion uniquement, pas de modification.)
        """
        suggestions: List[ReviewSuggestion] = []

        for folder, files in hierarchy.folder_structure.items():
            if len(files) != 1:
                continue

            if "/" not in folder:
                continue
            parent = folder.rsplit("/", 1)[0]

            suggestions.append(
                ReviewSuggestion(
                    action="merge_folder",
                    target=folder,
                    suggestion=(
                        f"Le dossier '{folder}' contient un seul fichier. "
                        f"Suggestion: fusionner dans '{parent}' pour réduire la fragmentation."
                    ),
                )
            )

        return suggestions

    def _suggest_review_folder_for_low_confidence(
        self,
        hierarchy: HierarchyProposal,
        cat_by_name: Dict[str, CategorizationResult],
    ) -> List[ReviewSuggestion]:
        """
        Suggère de router les fichiers low-confidence vers un dossier de revue.
        IMPORTANT: on ne change pas file_to_folder; c'est une suggestion.
        """
        low_files = []
        for fn in hierarchy.file_to_folder.keys():
            c = cat_by_name.get(fn)
            if c and float(c.confidence) < self.low_confidence_threshold:
                low_files.append(fn)

        if not low_files:
            return []

        return [
            ReviewSuggestion(
                action="move_file",
                target="Review/",
                suggestion=(
                    f"{len(low_files)} fichiers ont une faible confiance. "
                    "Suggestion: les placer dans un dossier 'Review/' pour vérification manuelle. "
                    f"Exemples: {', '.join(sorted(low_files)[:5])}"
                ),
            )
        ]

    def _suggest_rename_suspicious_folders(self, hierarchy: HierarchyProposal) -> List[ReviewSuggestion]:
        """
        Heuristique simple: détecter dossiers avec noms bizarres (double spaces, trailing underscores, etc.)
        et suggérer un renommage "nettoyage".
        """
        suggestions: List[ReviewSuggestion] = []
        for folder in hierarchy.folder_structure.keys():
            if "  " in folder or folder.endswith("_") or folder.endswith("-"):
                cleaned = re.sub(r"\s+", " ", folder).strip()
                cleaned = cleaned.rstrip("_-")
                if cleaned != folder:
                    suggestions.append(
                        ReviewSuggestion(
                            action="rename_folder",
                            target=folder,
                            suggestion=f"Renommer '{folder}' en '{cleaned}' (nettoyage format).",
                        )
                    )
        return suggestions
    
    
    def _llm_rewrite_suggestions(
        self,
        suggestions: List[ReviewSuggestion],
        hierarchy: HierarchyProposal,
    ) -> List[ReviewSuggestion]:
        """
        Reformule les suggestions via Ollama HTTP (LOCAL).
        - Pas de décision
        - Pas de changement action/target
        """
        import json
        import httpx

        prompt = (
            "Tu es un assistant chargé de reformuler des suggestions techniques.\n"
            "Tu NE DOIS PAS modifier leur sens.\n"
            "Tu NE DOIS PAS inventer de nouvelles suggestions.\n"
            "Tu NE DOIS PAS changer les actions ni les cibles.\n\n"
            "Reformule chaque suggestion pour qu'elle soit claire, concise et professionnelle.\n\n"
            "Réponds STRICTEMENT au format JSON :\n"
            "{ \"suggestions\": [ {\"suggestion\": \"...\"} ] }\n\n"
            f"SUGGESTIONS:\n{json.dumps([s.suggestion for s in suggestions], ensure_ascii=False, indent=2)}"
        )

        try:
            resp = httpx.post(
                f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.0,
                },
                timeout=120,
            )

            raw = resp.json().get("response", "")
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))

            rewritten = data.get("suggestions", [])
            if len(rewritten) != len(suggestions):
                return suggestions  # sécurité

            out = []
            for old, new in zip(suggestions, rewritten):
                out.append(
                    ReviewSuggestion(
                        action=old.action,
                        target=old.target,
                        suggestion=new["suggestion"],
                    )
                )
            return out

        except Exception as e:
            logger.warning("LLM Reviewer skipped (error): %s", e)
            return suggestions



# ======================================================
# Helper CLI / debug (optional)
# ======================================================
def print_review(review: ReviewResult) -> None:
    print("\n=== REVIEW ISSUES ===")
    for issue in review.issues:
        print(f"- [{issue.severity.upper()}] {issue.issue_type}")
        print(f"  {issue.description}")
        if issue.affected_files:
            print(f"  Files: {issue.affected_files}")

    print("\n=== REVIEW SUGGESTIONS ===")
    for sug in review.suggestions:
        print(f"- Action: {sug.action}")
        print(f"  Target: {sug.target}")
        print(f"  Suggestion: {sug.suggestion}")
