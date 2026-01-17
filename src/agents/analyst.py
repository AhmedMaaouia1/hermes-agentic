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

Implémentation de l’Agent Analyst (Agent 1) pour HERMES Agentic.

Responsabilité :
- Analyse sémantique locale d’un fichier unique
- Enrichissement du parsing via hypothèses légères et explicables
- Production d’un FileProfile STRICTEMENT conforme à core/types.py
"""

from typing import Dict, Any, Optional, Tuple
import os
import json

from core.types import ParsedFile, FileProfile
from core.config import ANALYST_USE_CONTENT, ANALYST_MAX_TOKENS

# --- LLM (local via Ollama) ---
from langchain_ollama import OllamaLLM

# ======================================================
# Prompt LLM (strictement contrôlé)
# ======================================================
ANALYST_PROMPT = """
Tu es un assistant NLP chargé d’analyser UN fichier individuellement.

À partir des informations fournies, tu dois produire :
- topic : hypothèse sémantique courte (1–3 mots)
- keywords : liste de mots-clés utiles (5–10 max)
- summary : 1 à 2 phrases factuelles, descriptives, sans interprétation

Règles STRICTES :
- Ne mentionne PAS de catégorie ou taxonomie
- Ne dis PAS où ranger le fichier
- Ne fais PAS d’évaluation ("important", "personnel", etc.)
- Reste factuel et neutre
- Si le contenu est pauvre ou absent, base-toi uniquement sur le nom et l’extension
- Si le texte ressemble à un énoncé académique, scolaire ou mathématique,
  décris-le comme tel sans inférer de domaine applicatif réel.


Réponds UNIQUEMENT au format JSON suivant :

{
  "topic": "...",
  "keywords": ["...", "..."],
  "summary": "..."
}
"""


# ======================================================
# Helpers contenu (LOCAL)
# ======================================================
def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    toks = text.split()
    return " ".join(toks[:max_tokens])


def _extract_txt(file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = {"content_source": "txt", "extraction_method": "builtin"}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), meta
    except Exception as e:
        meta["extraction_error"] = str(e)
        return None, meta


def _extract_pdf_text(file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = {"content_source": "pdf_text", "extraction_method": "pypdf2"}
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(file_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(t for t in pages if t.strip()).strip()
        if text:
            return text, meta
    except Exception as e:
        meta["pypdf2_error"] = str(e)

    meta_fb = {"content_source": "pdf_text", "extraction_method": "pdfplumber"}
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(file_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(t for t in pages if t.strip()).strip()
        return (text, meta_fb) if text else (None, meta_fb)
    except Exception as e:
        meta_fb["pdfplumber_error"] = str(e)
        return None, meta_fb


def _extract_image_ocr(file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = {
        "content_source": "ocr_text",
        "extraction_method": "tesseract+preprocessing",
        "ocr_success": False,
    }

    try:
        from PIL import Image, ImageFilter  # type: ignore
        import pytesseract  # type: ignore

        # --- 1. Chargement + niveaux de gris ---
        img = Image.open(file_path).convert("L")

        # --- 2. Amélioration du contraste (netteté légère) ---
        img = img.filter(ImageFilter.SHARPEN)

        # --- 3. Binarisation (seuillage simple) ---
        # seuil empirique robuste pour documents imprimés
        img = img.point(lambda x: 0 if x < 150 else 255, "1")

        # --- 4. OCR avec configuration adaptée aux documents ---
        text = pytesseract.image_to_string(
            img,
            lang="fra+eng",
            config="--psm 6"
        ).strip()

        if text:
            meta["ocr_success"] = True
            meta["content_tokens"] = len(text.split())
            return text, meta

        # OCR exécuté mais aucun texte exploitable
        meta["content_source"] = "image_no_text"
        meta["content_tokens"] = 0
        return None, meta

    except Exception as e:
        meta["ocr_error"] = str(e)
        meta["content_source"] = "image_no_text"
        meta["content_tokens"] = 0
        return None, meta


def _extract_content(
    parsed_file: ParsedFile, file_path: str, max_tokens: int
) -> Tuple[Optional[str], Dict[str, Any]]:
    ext = parsed_file.extension.lower()
    if ext == ".txt":
        text, meta = _extract_txt(file_path)
    elif ext == ".pdf":
        text, meta = _extract_pdf_text(file_path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        text, meta = _extract_image_ocr(file_path)
    else:
        return None, {"content_source": "unsupported", "extraction_method": None}

    if text:
        text = _truncate_text_by_tokens(text, max_tokens)
    return text, meta


# ======================================================
# Agent Analyst
# ======================================================
class AnalystAgent:
    def __init__(self, model_name: str = "phi3:mini"):
        self.llm = OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )


    def analyze(self, parsed_file: ParsedFile, file_path: Optional[str] = None) -> FileProfile:
        signals: Dict[str, Any] = {"source": "filename", "used_content": False}

        content_text: Optional[str] = None

        if ANALYST_USE_CONTENT and file_path and os.path.isfile(file_path):
            content_text, meta = _extract_content(parsed_file, file_path, ANALYST_MAX_TOKENS)
            signals.update(meta)

            signals["content_tokens"] = len(content_text.split()) if content_text else 0

            if content_text:
                signals["used_content"] = True
                signals["source"] = "content"

        # --------------------------------------------------
        # Cas image sans texte (sans LLM)
        # --------------------------------------------------
        if ANALYST_USE_CONTENT and signals.get("content_source") == "image_no_text":
            clean_tokens = [
                t for t in parsed_file.tokens
                if len(t) >= 3 and not t.isdigit()
            ]
            return FileProfile(
                filename=parsed_file.filename,
                file_type=parsed_file.file_type,
                topic="image_generic",
                keywords=clean_tokens[:10],
                summary="Image sans texte détectable, probablement une photo ou capture d’écran.",
                signals=signals,
            )

        # --------------------------------------------------
        # Décision d’appel LLM
        # --------------------------------------------------
        use_llm = ANALYST_USE_CONTENT or len(parsed_file.tokens) >= 2
        clean_tokens = [
            t for t in parsed_file.tokens
            if len(t) >= 3 and not t.isdigit()
        ]

        if not use_llm:
            return FileProfile(
                filename=parsed_file.filename,
                file_type=parsed_file.file_type,
                topic=None,
                keywords=clean_tokens[:5],
                summary=f"Le fichier nommé {parsed_file.filename} est un fichier de type {parsed_file.file_type}.",
                signals=signals,
            )

        # --------------------------------------------------
        # Appel LLM (input JSON structuré)
        # --------------------------------------------------
        llm_input = {
            "filename": parsed_file.filename,
            "extension": parsed_file.extension,
            "tokens": parsed_file.tokens,
            "file_type": parsed_file.file_type,
            "content_excerpt": content_text[:1500] if content_text else None,
        }

        prompt = ANALYST_PROMPT + "\n\nINPUT:\n" + json.dumps(llm_input, ensure_ascii=False)

        try:
            response = self.llm.invoke(prompt)
            result = self._safe_parse_llm_output(response)
        except Exception as e:
            signals["llm_error"] = str(e)
            result = {
                "topic": None,
                "keywords": parsed_file.tokens[:5],
                "summary": f"Le fichier nommé {parsed_file.filename} est un fichier de type {parsed_file.file_type}.",
            }

        return FileProfile(
            filename=parsed_file.filename,
            file_type=parsed_file.file_type,
            topic=result.get("topic"),
            keywords=result.get("keywords", []),
            summary=result.get("summary"),
            signals=signals,
        )

    # --------------------------------------------------
    # Sécurité JSON
    # --------------------------------------------------
    def _safe_parse_llm_output(self, output: str) -> Dict[str, Any]:
        """
        Parse robuste du JSON retourné par le LLM.
        Tente d'extraire le premier objet JSON valide sans interprétation.
        """
        import json
        import re

        # Extraction du premier bloc JSON détectable
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM output")

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM output contains invalid JSON") from exc


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
