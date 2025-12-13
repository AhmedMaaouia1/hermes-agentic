""" Code juste pour montrer comment on va utiliser les logs dans les agents """
from core.config import get_logger

logger = get_logger("AgentAnalyst")

def analyze_file(file_info):
    logger.info(f"Analyzing file: {file_info.filename}")
    # traitement
    logger.info("Analysis completed")
