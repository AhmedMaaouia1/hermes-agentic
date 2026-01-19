import logging
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import json
import subprocess

from core.types import (
    ParsedFile,
    FileProfile,
    CategorizationResult,
    HierarchyProposal,
    ReviewResult,
    ReviewIssue,
    ReviewSuggestion,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReviewerAgent:
    def __init__(
        self,
        llm_model: str = "llama3:8b",
        llm_temperature: float = 0.0,
        confidence_threshold: float = 0.4,
    ):
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.confidence_threshold = confidence_threshold

    def review(
        self,
        parsed_files: List[ParsedFile],
        profiles: List[FileProfile],
        categorizations: List[CategorizationResult],
        hierarchy: HierarchyProposal,
    ) -> ReviewResult:
        logger.info("Starting hierarchy review")

        issues: List[ReviewIssue] = []
        suggestions: List[ReviewSuggestion] = []

        profile_map = {p.file_id: p for p in profiles}
        categ_map = {c.file_id: c for c in categorizations}

        issues.extend(
            self._detect_low_confidence_categorizations(categorizations)
        )
        issues.extend(
            self._detect_profile_category_mismatch(profile_map, categ_map)
        )
        issues.extend(
            self._detect_heterogeneous_folders(hierarchy, categ_map)
        )

        llm_suggestions = self._llm_semantic_review(hierarchy)
        suggestions.extend(llm_suggestions)

        logger.info(
            "Review completed: %d issues, %d suggestions",
            len(issues),
            len(suggestions),
        )

        return ReviewResult(
            issues=issues,
            suggestions=suggestions,
            revised_structure=None,
        )

    def _detect_low_confidence_categorizations(
        self, categorizations: List[CategorizationResult]
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for cat in categorizations:
            if cat.confidence < self.confidence_threshold:
                issues.append(
                    ReviewIssue(
                        issue_type="low_confidence_categorization",
                        severity="medium",
                        description=(
                            f"Low confidence ({cat.confidence:.2f}) "
                            f"for category '{cat.category}'."
                        ),
                        affected_files=[cat.file_id],
                    )
                )
        return issues

    def _detect_profile_category_mismatch(
        self,
        profiles: Dict[str, FileProfile],
        categs: Dict[str, CategorizationResult],
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for file_id, profile in profiles.items():
            if file_id not in categs:
                continue

            cat = categs[file_id]
            if hasattr(profile, "topics") and profile.topics:
                if cat.category not in profile.topics:
                    issues.append(
                        ReviewIssue(
                            issue_type="semantic_mismatch",
                            severity="medium",
                            description=(
                                f"Category '{cat.category}' is weakly aligned "
                                f"with detected topics {profile.topics}."
                            ),
                            affected_files=[file_id],
                        )
                    )
        return issues

    def _detect_heterogeneous_folders(
        self,
        hierarchy: HierarchyProposal,
        categs: Dict[str, CategorizationResult],
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for folder, file_ids in hierarchy.structure.items():
            categories = [
                categs[f].category
                for f in file_ids
                if f in categs
            ]
            unique_categories = set(categories)

            if len(unique_categories) > 3:
                issues.append(
                    ReviewIssue(
                        issue_type="heterogeneous_folder",
                        severity="high",
                        description=(
                            f"Folder '{folder}' contains multiple unrelated "
                            f"categories: {sorted(unique_categories)}."
                        ),
                        affected_files=file_ids,
                    )
                )
        return issues

    def _llm_semantic_review(
        self, hierarchy: HierarchyProposal
    ) -> List[ReviewSuggestion]:
        prompt = self._build_llm_prompt(hierarchy)
        raw_output = self._call_llm(prompt)

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("LLM output could not be parsed as JSON")
            return []

        suggestions: List[ReviewSuggestion] = []

        for item in data.get("suggestions", []):
            try:
                suggestions.append(
                    ReviewSuggestion(
                        action=item["action"],
                        source=item.get("source"),
                        target=item.get("target"),
                        reason=item.get("reason", ""),
                    )
                )
            except KeyError:
                continue

        return suggestions

    def _build_llm_prompt(self, hierarchy: HierarchyProposal) -> str:
        return (
            "You are a critical reviewer of a file hierarchy.\n"
            "Analyze the following folder structure and suggest ONLY:\n"
            "- clearer folder names\n"
            "- possible folder merges\n\n"
            "Do NOT invent new categories.\n"
            "Do NOT apply changes.\n"
            "Return STRICT JSON with a 'suggestions' list.\n\n"
            f"Hierarchy:\n{json.dumps(hierarchy.structure, indent=2)}"
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                [
                    "ollama",
                    "run",
                    self.llm_model,
                ],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.stdout.decode("utf-8").strip()
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""
    def print_review(review: ReviewResult):
        print("\n=== REVIEW ISSUES ===")
        for issue in review.issues:
            print(f"- [{issue.severity.upper()}] {issue.issue_type}")
            print(f"  {issue.description}")
            print(f"  Files: {issue.affected_files}")

        print("\n=== REVIEW SUGGESTIONS ===")
        for sug in review.suggestions:
            print(f"- Action: {sug.action}")
            print(f"  From: {sug.source} → To: {sug.target}")
            print(f"  Reason: {sug.reason}")