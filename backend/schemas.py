from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class FlowResult(BaseModel):
    flow_index: int
    prediction: str
    confidence: Optional[float] = None

class ResultsResponse(BaseModel):
    job_id: str
    total_flows: int
    results: List[FlowResult]

class StatsResponse(BaseModel):
    job_id: str
    total_flows: int
    counts_by_label: Dict[str, int]
    attack_percentage: float