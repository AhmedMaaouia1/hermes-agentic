# Interface Web Locale HERMES Agentic

Ce document décrit l'architecture de l'interface web locale et explique comment la lancer.

## 1. Architecture des fichiers

```
src/
  web/
    app.py                # Backend FastAPI
    static/
      index.html          # Frontend
      styles.css
      app.js
```

## 2. Backend Python (FastAPI)

- **Point d'entrée** : `src/web/app.py`
- **Endpoints** :
  - `GET /` → sert l'interface HTML
  - `GET /health` → vérification rapide
  - `POST /run_pipeline` → exécute `run_pipeline(folder_path)` et retourne un JSON structuré

### Format JSON retourné

```
{
  "status": "success",
  "data": {
    "hierarchy": {
      "folder_structure": { ... },
      "file_to_folder": { ... },
      "rationale": "...",
      "warnings": [ ... ]
    },
    "categorizations": [
      {
        "filename": "...",
        "category": "...",
        "subcategory": "...",
        "confidence": 0.92,
        "rationale": "...",
        "decision_source": "..."
      }
    ],
    "reviewer": {
      "issues": [ ... ],
      "suggestions": [ ... ],
      "revised_structure": null
    }
  }
}
```

## 3. Frontend minimal fonctionnel

Le frontend est un HTML/CSS/JS simple :

- Sélecteur de dossier via `input webkitdirectory`
- Champ texte pour le chemin absolu (nécessaire pour appeler le pipeline local)
- Bouton "Analyser" avec indicateur de chargement
- Résumé rapide (volumétrie, confiance faible, avertissements, doublons potentiels)
- Tree view pour la hiérarchie
- Couleurs de confiance (vert / orange / rouge)
- Avertissements et suggestions clairement séparés

## 4. Points d'intégration avec le pipeline existant

Le backend appelle directement :

```python
from orchestration.pipeline import run_pipeline
```

Le endpoint `/run_pipeline` exécute `run_pipeline(folder_path)` et renvoie les objets Pydantic sérialisés (`model_dump`).

## 5. Lancement en local

### Sans Docker

```bash
pip install -r requirements.txt
uvicorn web.app:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

Puis ouvrir : `http://localhost:8000`

### Avec Docker (exemple générique)

```bash
docker build -t hermes-agentic-web .
docker run --rm -p 8000:8000 -v "$(pwd)":/app hermes-agentic-web \
  uvicorn web.app:app --app-dir src --host 0.0.0.0 --port 8000
```

> Ajustez le `Dockerfile`/`docker-compose` existant si besoin pour exposer le port 8000.
