import os
from typing import Dict, Any, List, Optional
from core.config import get_logger
from core.types import ParsedFile, FileProfile

# Extraction libs
try:
    from pypdf import PdfReader
    import docx
except ImportError:
    PdfReader = None
    docx = None

logger = get_logger("AgentAnalyst")


def extract_content(file_path: str, file_type: str) -> str:
    """
    Extrait le texte brut des 1000 premiers caractères d'un fichier PDF, DOCX ou TXT.
    """
    text = ""
    try:
        if file_type == "document":
            if file_path.lower().endswith(".pdf") and PdfReader:
                reader = PdfReader(file_path)
                if len(reader.pages) > 0:
                    text = reader.pages[0].extract_text() or ""
            
            elif file_path.lower().endswith(".docx") and docx:
                doc = docx.Document(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                    if len(full_text) > 10: # Limit reading
                        break
                text = "\n".join(full_text)

        if file_type == "code" or file_path.lower().endswith(".txt"):
             with open(file_path, 'r', errors='ignore') as f:
                 text = f.read(1000)

    except Exception as e:
        logger.warning(f"Failed to extract content from {file_path}: {e}")
    
    return text.lower()


def analyze_file(parsed_file: ParsedFile, file_path: str = None) -> FileProfile:
    """
    Analyse un fichier pour en déduire un profil sémantique.
    """
    logger.info(f"Analyzing file: {parsed_file.filename}")

    # 1. Extraction du contenu
    content_snippet = ""
    if file_path and os.path.exists(file_path):
        content_snippet = extract_content(file_path, parsed_file.file_type)

    topic = "unknown"
    name_tokens = set(parsed_file.tokens)
    
    import re
    # helper
    def matches(keywords, source_text):
        if isinstance(source_text, (list, set)):
            source_text = " ".join(source_text)
        
        # Build regex for word boundaries for short keywords (< 4 chars)
        # For longer keywords, substring match is usually fine, but let's be safe.
        # Actually simplest is: if k is short, use boundary, else substring?
        # Let's simple check:
        for k in keywords:
            if len(k) < 4:
                # Regex boundary search
                if re.search(r'\b' + re.escape(k) + r'\b', source_text):
                    return True
            else:
                # Substring match
                if k in source_text:
                    return True
        return False

    # 2. Détection avancée par Topic (Granularité Subcategory)
    # Use normalized_name for filename matching to handle "machine_learning" -> "machine learning"
    filename_search_space = parsed_file.normalized_name.replace("_", " ").lower()
    
    topics_map = {
        # Administratif
        "facture": ["facture", "invoice", "recu", "ticket", "paiement", "payment", "total", "ttc"],
        "impots": ["impot", "tax", "declaration", "fisc", "revenu", "avis"],
        "identite": ["passport", "passeport", "identite", "identity", "cni", "national"],
        "cv": ["cv", "resume", "curriculum", "experience", "education", "skills"],
        "attestation": ["attestation", "certificat", "scolarite", "certificate", "school"],
        "contrat": ["contrat", "avenant"],

        # Cours
        "cours_nlp": ["nlp", "natural language", "linguistique", "transformer", "bert", "gpt"],
        "cours_ml": ["machine learning", "apprentissage", "neural", "deep learning", "regression", "svm"],
        "cours_ia": ["intelligence artificielle", "ai", "ia", "artificial"],
        
        # Projets
        "rapport_projet": ["rapport", "report", "soutenance", "memoire", "pfe", "stage"],
        
        # Images/Scans (explicit topic to catch pdf scans)
        "scan": ["scan", "numerisation"],

        # Code (souvent extension, mais keywords aussi)
        "code_source": ["def ", "import ", "class ", "function", "var ", "const "],
    }

    # Logique de détection
    detected = None

    # Iteration sur les topics prioritaires
    # Ordre d'evaluation important ? On test tout
    
    for t_name, keywords in topics_map.items():
        if matches(keywords, filename_search_space):
            detected = t_name
            break # Filename win
        elif matches(keywords, content_snippet):
            detected = t_name
            # On continue pour voir si filename match autre chose? non, content est fort aussi.
            # Mais attention aux faux positifs dans le texte.
            # Pour l'instant first mach.
            break

    if detected:
        topic = detected
    else:
        # Fallback génériques
        if parsed_file.file_type == "image":
            topic = "image"
        elif parsed_file.file_type == "code":
            topic = "code_source"
        elif parsed_file.file_type == "archive":
            topic = "archive"
        elif parsed_file.file_type == "executable":
            topic = "executable"
        elif parsed_file.extension in [".xlsx", ".xls", ".csv"]:
            topic = "data_table"

    # 3. Construction result
    ignored = {"le", "la", "les", "de", "du", "et", "en", "a", "of", "the", "and"}
    keywords = [t for t in parsed_file.tokens if len(t) > 2 and t not in ignored]

    signals = {
        "has_year": parsed_file.has_year,
        "extension": parsed_file.extension,
        "content_len": len(content_snippet)
    }

    profile = FileProfile(
        filename=parsed_file.filename,
        file_type=parsed_file.file_type,
        topic=topic,
        keywords=keywords,
        signals=signals
    )

    logger.debug(f"Analysis result for {parsed_file.filename}: {topic}")
    return profile
