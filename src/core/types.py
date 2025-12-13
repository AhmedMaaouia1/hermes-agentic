from pydantic import BaseModel
from typing import Optional, List

class FileInfo(BaseModel):
    filename: str
    extension: str
    created_at: Optional[str]
    modified_at: Optional[str]

class FileProfile(BaseModel):
    filename: str
    type: str
    topic: str

class Categorization(BaseModel):
    filename: str
    category: str
    subcategory: Optional[str]
    confidence: float
    rationale: str
