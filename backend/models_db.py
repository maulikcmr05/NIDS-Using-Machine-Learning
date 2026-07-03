import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from database import Base

class UploadJob(Base):
    __tablename__ = "upload_jobs"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename     = Column(String)
    file_type    = Column(String)
    status       = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    results_path = Column(String, nullable=True)