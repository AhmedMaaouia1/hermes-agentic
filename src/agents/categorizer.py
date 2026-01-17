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
