from agents.planner import PlannerAgent
from core.types import CategorizationResult

agent = PlannerAgent()

fake_data = [
    CategorizationResult(
        filename="facture_2023.pdf",
        category="Administratif",
        subcategory="Factures",
        confidence=0.92,
        rationale="Invoice document"
    ),
    CategorizationResult(
        filename="cours_nlp.pdf",
        category="Cours",
        subcategory="NLP",
        confidence=0.55,
        rationale="Course material"
    ),
]

proposal = agent.plan(fake_data)

print(proposal)
