import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from job_processor import process_upload
from models_db import UploadJob
from schemas import (
    UploadResponse, JobStatusResponse,
    ResultsResponse, FlowResult, StatsResponse,
)

# Create database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="NIDS Backend API")

# Allow frontend (person 4) to call this API from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "NIDS Backend is running. Visit /docs for the API."}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Only accept CSV files
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save the job record to the database
    job = UploadJob(filename=file.filename, file_type="csv")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Save the uploaded file to disk
    dest = UPLOAD_DIR / f"{job.id}.csv"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Start processing in the background (so we return immediately)
    background_tasks.add_task(process_upload, job.id, str(dest))

    return UploadResponse(job_id=job.id, filename=file.filename, status=job.status)


@app.get("/results/{job_id}/status", response_model=JobStatusResponse)
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
def get_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed yet. Status: {job.status}")

    df = pd.read_csv(job.results_path)
    results = [FlowResult(**row) for row in df.to_dict(orient="records")]
    return ResultsResponse(job_id=job_id, total_flows=len(results), results=results)


@app.get("/results/{job_id}/stats", response_model=StatsResponse)
def get_stats(job_id: str, db: Session = Depends(get_db)):
    job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
    if not job or job.status != "completed":
        raise HTTPException(status_code=404, detail="Results not available yet.")

    df = pd.read_csv(job.results_path)
    counts = df["prediction"].value_counts().to_dict()
    total  = len(df)
    benign = counts.get("BENIGN", 0)
    attack_pct = round((total - benign) / total * 100, 2) if total else 0.0

    return StatsResponse(
        job_id=job_id,
        total_flows=total,
        counts_by_label=counts,
        attack_percentage=attack_pct,
    )


@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    jobs = db.query(UploadJob).order_by(UploadJob.created_at.desc()).all()
    return [
        {
            "job_id":     j.id,
            "filename":   j.filename,
            "status":     j.status,
            "created_at": j.created_at,
        }
        for j in jobs
    ]