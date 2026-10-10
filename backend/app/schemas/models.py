
from pydantic import BaseModel, Field
from typing import Optional, Any

class RecommendationRequest(BaseModel):
    show_id: str
    limit: int = Field(default=8, ge=1, le=30)
    type_filter: Optional[str] = None
    genre_filter: Optional[str] = None

class TrainRequest(BaseModel):
    task: str
    algorithm: str

class PredictRequest(BaseModel):
    task: str
    algorithm: str
    values: dict[str, Any]
