from core.config import get_logger
from core.types import FileProfile, CategorizationResult

logger = get_logger("AgentCategorizer")

TOPIC_MAP = {
    # Administratif
    "facture": ("Administratif", "Factures", 0.95),
    "impots": ("Administratif", "Impots", 0.95),
    "identite": ("Administratif", "Identite", 0.90),
    "cv": ("Administratif", "CV", 0.95),
    "attestation": ("Administratif", "Attestations", 0.90),
    "contrat": ("Administratif", "Contrats", 0.90),
    "scan": ("Images", "Scans", 0.90),

    # Cours
    "cours_nlp": ("Cours", "NLP", 0.90),
    "cours_ml": ("Cours", "Machine_Learning", 0.90),
    "cours_ia": ("Cours", "IA", 0.85),
    "cours": ("Cours", "Divers", 0.70),

    # Projets
    "rapport_projet": ("Projets", "Rapports", 0.80),
    "projet": ("Projets", "IA", 0.60),

    # Technique
    "code_source": ("Projets", "Code", 0.85),
    "archive": ("Archives", "ZIP", 0.80),
    "executable": ("Logiciels", "Installateurs", 0.80),
    "data_table": ("Donnees", "Tableurs", 0.80),
}

def categorize_file(profile: FileProfile) -> CategorizationResult:
    """
    Categorize a file based on its topic according to the taxonomy (Noussayba).
    """
    logger.info(f"Categorizing file: {profile.filename} (Topic: {profile.topic})")

    t = profile.topic
    category, subcategory, confidence = TOPIC_MAP.get(
        t, ("Divers", None, 0.5)
    )

    # Traitement spécial pour les images
    if t == "image":
        category = "Images"
        fname = profile.filename.lower()
        if "screenshot" in fname or "capture" in fname:
            subcategory = "Screenshots"
        elif "scan" in fname:
            subcategory = "Scans"
        else:
            subcategory = "Photos"
        confidence = 0.8

    result = CategorizationResult(
        filename=profile.filename,
        category=category,
        subcategory=subcategory,
        confidence=confidence,
        rationale=f"Mapped from topic '{t}'",
        decision_source="taxonomy_rules_v2"
    )

    logger.debug(f"Categorization: {result.category}/{result.subcategory}")
    return result
"""
HERMES Agentic — Agent Catégorisation (Agent 2)
===============================================

Responsabilité UNIQUE :
----------------------
À partir d'un FileProfile (Agent Analyst),
assigner (category, subcategory, confidence, rationale, decision_source)
STRICTEMENT issus de src/resources/taxonomy.json.

Contraintes académiques STRICTES :
----------------------------------
- Taxonomie fermée : taxonomy.json = SEULE source de vérité
- Décision fichier par fichier (aucun raisonnement global, aucun clustering)
- Règles déterministes prioritaires (haute précision, explicables)
- Embeddings UNIQUEMENT en fallback (fichier ↔ descriptions taxonomie)
- Ne jamais inventer de catégorie/sous-catégorie
- Sorties conformes à src/core/types.py
- Logging systématique : rule_id / embedding_score / fallback

Stratégie :
-----------
1) Rules Engine
2) Embedding fallback (lazy-loaded)
3) Si similarité faible → catégorie "Divers" (si présente), confidence basse
"""

import json
import logging
import os
from typing import Dict, Optional, List, Tuple, Any


# =========================
# Imports types (robustes)
# =========================
# Selon votre convention d'exécution (PYTHONPATH / module), l'un des imports marchera.
try:
    from core.types import FileProfile, CategorizationResult  # convention "core.*"
except ModuleNotFoundError:
    from src.core.types import FileProfile, CategorizationResult  # convention "src.core.*"


# =========================
# Logger
# =========================
logger = logging.getLogger("hermes.categorizer")


# =========================
# Utilitaires
# =========================

def clean_keywords(keywords: List[str]) -> List[str]:
    """
    Nettoyage léger, déterministe et non sémantique des keywords issus du LLM Analyst.

    Autorisé :
    - lowercase/strip
    - suppression doublons
    - filtrage longueur
    - stopwords basiques

    Interdit :
    - enrichissement sémantique
    - correction "intelligente"
    """
    stopwords = {
        "file", "document", "documents", "pdf", "image", "img", "data",
        "fichier", "texte", "text", "copy", "copie"
    }

    cleaned: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        kw = kw.strip().lower()
        if len(kw) < 3:
            continue
        if kw in stopwords:
            continue
        cleaned.append(kw)

    return sorted(set(cleaned))


def build_profile_text(profile: FileProfile, keywords: List[str]) -> str:
    """
    Construit la représentation textuelle autorisée du fichier :
    filename + topic + keywords + summary (si dispo).

    Aucun contenu brut n'est utilisé.
    """
    parts = [
        profile.filename,
        profile.topic or "",
        " ".join(keywords),
        profile.summary or ""
    ]
    return " ".join(p for p in parts if p).strip()


def get_extension(filename: str) -> str:
    """Retourne l'extension (lowercase), ex: '.pdf' ou '' si absent."""
    _, ext = os.path.splitext(filename)
    return (ext or "").lower().strip()


def sim_to_conf(sim: float) -> float:
    """
    Calibration conservative similarity (cosine) -> confidence interprétable.
    Objectif : éviter 'cosine == probabilité', plus défendable académiquement.
    """
    # clamp
    sim = max(-1.0, min(1.0, float(sim)))

    # (valeurs typiques sur MiniLM : bons matchs ~0.50-0.75 selon domaine)
    if sim < 0.35:
        return 0.40
    if sim < 0.50:
        return 0.55
    if sim < 0.65:
        return 0.70
    return 0.80


# =========================
# Agent Catégorisation
# =========================

class CategorizerAgent:
    """
    Agent 2 — Catégorisation sémantique locale

    Entrée : FileProfile
    Sortie : CategorizationResult
    """

    def __init__(
        self,
        taxonomy_path: str = "src/resources/taxonomy.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = True
    ):
        # --- Source de vérité : TAXONOMY.JSON
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            # schema attendu: {category: {subcategory: {"description": "..."} } } ou { "Divers": {} }
            self.taxonomy: Dict[str, Dict[str, Dict[str, str]]] = json.load(f)

        self.taxonomy_path = taxonomy_path
        self.enable_embeddings = enable_embeddings
        self.embedding_model_name = embedding_model

        # --- Lazy-loading embeddings (optimisation propre)
        self._model = None
        self._taxon_embeddings: Optional[Dict[Tuple[str, str], Any]] = None  # (cat, sub) -> embedding

        logger.info(
            "CategorizerAgent initialized "
            f"(enable_embeddings={self.enable_embeddings}, taxonomy='{self.taxonomy_path}')."
        )

    # ==================================================
    # API publique
    # ==================================================

    def categorize(self, profile: FileProfile) -> CategorizationResult:
        """
        Point d'entrée principal.
        Ordre strict :
        1) règles
        2) embeddings fallback (si activé)
        3) fallback Divers (si embeddings désactivés ou indécidable)
        """
        keywords = clean_keywords(profile.keywords)

        # 1) Rules
        rule_result = self._apply_rules(profile, keywords)
        if rule_result:
            return rule_result

        # 2) Embeddings fallback
        if self.enable_embeddings:
            return self._embedding_fallback(profile, keywords)

        # 3) Si embeddings désactivés : Divers si existe sinon catégorie déterministe
        if "Divers" in self.taxonomy:
            return self._build_result(
                profile=profile,
                keywords=keywords,
                category="Divers",
                subcategory=None,
                confidence=0.40,
                source="rule",
                rule_id="fallback_no_embeddings",
                extra="embeddings_disabled_fallback"
            )

        # fallback ultime (déterministe)
        first_cat = sorted(self.taxonomy.keys())[0]
        sub = self._default_subcategory(first_cat)
        return self._build_result(
            profile=profile,
            keywords=keywords,
            category=first_cat,
            subcategory=sub,
            confidence=0.35,
            source="rule",
            rule_id="fallback_first_category",
            extra="no_rule_no_embeddings"
        )

    # ==================================================
    # Règles déterministes (prioritaires)
    # ==================================================

    def _apply_rules(self, profile: FileProfile, keywords: List[str]) -> Optional[CategorizationResult]:
        """
        Règles simples, explicables, haute précision.
        IMPORTANT : toute sortie doit exister dans taxonomy.json.
        """
        filename = profile.filename.lower()
        ext = get_extension(profile.filename)

        # Texte "signal" : keywords + topic + summary (si présent)
        # (toujours sans contenu brut)
        signal_text = " ".join(keywords)
        if profile.topic:
            signal_text += " " + profile.topic.lower()
        if profile.summary:
            signal_text += " " + profile.summary.lower()

        # -------------------------
        # Administratif (PDF + mots forts)
        # -------------------------
        if ext == ".pdf":
            if "facture" in signal_text or "tva" in signal_text or "reçu" in signal_text or "recu" in signal_text:
                return self._build_result(profile, keywords, "Administratif", "Factures", 0.95, "rule",
                                         rule_id="admin_facture_pdf")

            if "impot" in signal_text or "impôts" in signal_text or "taxe" in signal_text or "avis d" in signal_text:
                return self._build_result(profile, keywords, "Administratif", "Impots", 0.93, "rule",
                                         rule_id="admin_impots_pdf")

            if ("identite" in signal_text or "identité" in signal_text or "passeport" in signal_text
                    or "carte" in signal_text or "cin" in signal_text):
                return self._build_result(profile, keywords, "Administratif", "Identite", 0.94, "rule",
                                         rule_id="admin_identite_pdf")

            if "cv" in signal_text or "curriculum" in signal_text or "resume" in signal_text or "candidature" in signal_text:
                return self._build_result(profile, keywords, "Administratif", "CV", 0.93, "rule",
                                         rule_id="admin_cv_pdf")

            if "attestation" in signal_text or "certificat" in signal_text:
                return self._build_result(profile, keywords, "Administratif", "Attestations", 0.92, "rule",
                                         rule_id="admin_attestation_pdf")

        # -------------------------
        # Données
        # -------------------------
        if ext in (".xlsx", ".csv"):
            return self._build_result(profile, keywords, "Donnees", "Tableurs", 0.95, "rule",
                                     rule_id="data_tableur")

        # -------------------------
        # Logiciels
        # -------------------------
        if ext == ".exe":
            return self._build_result(profile, keywords, "Logiciels", "Installateurs", 0.96, "rule",
                                     rule_id="logiciel_installateur")

        # -------------------------
        # Archives
        # -------------------------
        if ext == ".zip":
            return self._build_result(profile, keywords, "Archives", "ZIP", 0.92, "rule",
                                     rule_id="archive_zip")

        # -------------------------
        # Images (amélioration légère + OCR-aware)
        # -------------------------
        if ext in (".jpg", ".jpeg", ".png"):
            # Si du texte a été extrait par OCR de manière significative,
            # on ne force PAS une classification Image.
            # On laisse le fallback embeddings décider (ex: Cours/Divers).
            if (
                profile.signals.get("used_content") is True
                and profile.signals.get("content_tokens", 0) > 30
            ):
                return None

            # Heuristique simple : screenshot / scan / photo
            if "screenshot" in signal_text or "screen" in signal_text or "capture" in signal_text:
                return self._build_result(
                    profile, keywords,
                    "Images", "Screenshots",
                    0.90, "rule",
                    rule_id="image_screenshot"
                )

            if "scan" in signal_text or "scann" in signal_text:
                return self._build_result(
                    profile, keywords,
                    "Images", "Scans",
                    0.90, "rule",
                    rule_id="image_scan"
                )

            # Par défaut : photo
            return self._build_result(
                profile, keywords,
                "Images", "Photos",
                0.88, "rule",
                rule_id="image_photo"
            )


        # -------------------------
        # Projets (code / rapports) — seulement si signaux explicites
        # -------------------------
        # (Important : ici on reste minimal et précis ; embeddings gérera le reste)
        if ext in (".py", ".js", ".ts", ".java", ".c", ".cpp", ".ipynb"):
            # Ces extensions suggèrent "Projets/Code" mais seulement si présent dans taxonomie
            if self._exists("Projets", "Code"):
                return self._build_result(profile, keywords, "Projets", "Code", 0.90, "rule",
                                         rule_id="projet_code_by_ext")

        if ext in (".docx", ".pptx", ".ppt", ".pdf"):
            # Rapports : si "rapport" présent explicitement
            if "rapport" in signal_text and self._exists("Projets", "Rapports"):
                return self._build_result(profile, keywords, "Projets", "Rapports", 0.88, "rule",
                                         rule_id="projet_rapport_keyword")

        return None

    # ==================================================
    # Embedding fallback (lazy load + calibration + Divers fallback)
    # ==================================================

    def _embedding_fallback(self, profile: FileProfile, keywords: List[str]) -> CategorizationResult:
        """
        Compare FileProfile.text ↔ descriptions (cat/subcat) issues de taxonomy.json.

        Correctifs :
        - Divers fallback explicite hors embeddings (Option A)
        - sim_to_conf calibration (cosine != probabilité)
        - lazy loading modèle + embeddings taxonomie
        """
        self._ensure_embeddings_ready()

        query_text = build_profile_text(profile, keywords)
        query_emb = self._model.encode(query_text, normalize_embeddings=True)

        best_score = -1.0
        best_cat, best_sub = None, None

        for (cat, sub), emb in self._taxon_embeddings.items():
            # cosine similarity
            score = float(self._cos_sim(query_emb, emb))
            if score > best_score:
                best_score = score
                best_cat, best_sub = cat, sub

        confidence = sim_to_conf(best_score)

        # ---- Option A (recommandée) : Divers fallback explicite (cat-only)
        if confidence < 0.45 and "Divers" in self.taxonomy:
            logger.warning(
                f"[FALLBACK] file={profile.filename} reason=low_similarity "
                f"best={best_cat}/{best_sub} score={best_score:.3f} conf={confidence:.2f}"
            )
            return self._build_result(
                profile=profile,
                keywords=keywords,
                category="Divers",
                subcategory=None,
                confidence=0.40,
                source="embedding",
                rule_id=None,
                extra="low_similarity_fallback_to_Divers"
            )

        logger.info(
            f"[EMBEDDING] file={profile.filename} best={best_cat}/{best_sub} "
            f"score={best_score:.3f} conf={confidence:.2f}"
        )

        # sécurité : best_cat/sub doivent exister (normalement oui car taxon_embeddings vient de taxonomy)
        return self._build_result(
            profile=profile,
            keywords=keywords,
            category=best_cat,
            subcategory=best_sub,
            confidence=confidence,
            source="embedding",
            extra=f"similarity={best_score:.3f}"
        )

    # ==================================================
    # Embeddings: lazy init + precompute
    # ==================================================

    def _ensure_embeddings_ready(self) -> None:
        """Charge le modèle et précalcule les embeddings taxonomie au premier besoin."""
        if self._model is None:
            # Import ici pour éviter de charger inutilement si rules suffisent.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"[EMBEDDING] model_loaded={self.embedding_model_name}")

        if self._taxon_embeddings is None:
            self._taxon_embeddings = self._build_taxon_embeddings()
            logger.info(f"[EMBEDDING] taxonomy_embeddings_built n={len(self._taxon_embeddings)}")

    def _build_taxon_embeddings(self) -> Dict[Tuple[str, str], Any]:
        """
        Construit les embeddings pour chaque (category, subcategory) défini dans taxonomy.json
        en utilisant : "cat sub. description".

        IMPORTANT :
        - On ne construit PAS d'embedding pour une catégorie sans sous-catégorie (ex: Divers).
          Divers est traité comme fallback explicite (Option A).
        """
        emb_map: Dict[Tuple[str, str], Any] = {}

        for cat, subs in self.taxonomy.items():
            if not subs:
                continue  # ex: "Divers": {} -> pas de (cat, sub) ici
            for sub, meta in subs.items():
                desc = ""
                if isinstance(meta, dict):
                    desc = meta.get("description", "") or ""
                text = f"{cat} {sub}. {desc}".strip()
                emb_map[(cat, sub)] = self._model.encode(text, normalize_embeddings=True)

        return emb_map

    @staticmethod
    def _cos_sim(a_emb: Any, b_emb: Any) -> float:
        """Cosine similarity via sentence-transformers util.cos_sim, mais import minimal."""
        from sentence_transformers import util
        return util.cos_sim(a_emb, b_emb).item()

    # ==================================================
    # Validation taxonomie + build result + logging
    # ==================================================

    def _exists(self, category: str, subcategory: Optional[str]) -> bool:
        """Vérifie l'existence exacte (category, subcategory) dans taxonomy.json."""
        if category not in self.taxonomy:
            return False

        subs = self.taxonomy[category]
        # Catégorie sans sous-catégories (ex: Divers)
        if not subs:
            return subcategory is None

        # Catégorie avec sous-catégories => subcategory doit exister
        if subcategory is None:
            return False
        return subcategory in subs

    def _default_subcategory(self, category: str) -> Optional[str]:
        """Retourne une sous-catégorie déterministe si la catégorie en a, sinon None."""
        subs = self.taxonomy.get(category, {})
        if not subs:
            return None
        return sorted(subs.keys())[0]

    def _build_result(
        self,
        profile: FileProfile,
        keywords: List[str],
        category: str,
        subcategory: Optional[str],
        confidence: float,
        source: str,
        rule_id: Optional[str] = None,
        extra: Optional[str] = None
    ) -> CategorizationResult:
        """
        Construit un CategorizationResult en garantissant :
        - labels existants exactement dans taxonomy.json
        - rationale explicite (signaux utilisés)
        - logs cohérents
        """
        # --- Validation taxonomie (hard fail si violation)
        if not self._exists(category, subcategory):
            raise ValueError(
                f"INVALID TAXON OUTPUT: ({category}, {subcategory}) not in taxonomy.json"
            )

        label = category + (f"/{subcategory}" if subcategory else "")

        # --- Logging décision
        if source == "rule":
            logger.info(
                f"[RULE] file={profile.filename} rule_id={rule_id} "
                f"category={label} confidence={confidence:.2f}"
            )

        # --- Rationale explicable
        sig_parts = [f"filename='{profile.filename}'"]
        if profile.topic:
            sig_parts.append(f"topic='{profile.topic}'")
        if keywords:
            sig_parts.append(f"keywords_clean={keywords[:10]}")
        if profile.summary:
            sig_parts.append("summary_present")

        rationale = (
            f"Classé comme {label} via {source}. "
            f"Confiance={confidence:.2f}. "
            f"Signaux utilisés: {', '.join(sig_parts)}."
        )
        if rule_id:
            rationale += f" rule_id={rule_id}."
        if extra:
            rationale += f" {extra}"

        return CategorizationResult(
            filename=profile.filename,
            category=category,
            subcategory=subcategory,
            confidence=round(float(confidence), 3),
            rationale=rationale,
            decision_source=source
        )
