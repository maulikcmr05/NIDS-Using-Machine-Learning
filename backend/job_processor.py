from datetime import datetime
from pathlib import Path
import pandas as pd

from database import SessionLocal
from data_utils import align_prediction_columns
from model_handler import model_handler
from models_db import UploadJob

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def process_upload(job_id: str, file_path: str):
    db = SessionLocal()
    try:
        # Mark as processing
        job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
        job.status = "processing"
        db.commit()

        # Step 1: Read the uploaded CSV
        df = pd.read_csv(file_path, low_memory=False)

        # Step 2: Align columns exactly as person 2's training pipeline did
        processed = align_prediction_columns(df, model_handler.feature_columns)

        # Step 3: Run the model
        labels, confidences = model_handler.predict(processed)

        # Step 4: Save results to a CSV
        results_df = pd.DataFrame({
            "flow_index": range(len(labels)),
            "prediction": labels,
            "confidence": confidences if confidences is not None else [None] * len(labels),
        })
        results_path = RESULTS_DIR / f"{job_id}.csv"
        results_df.to_csv(results_path, index=False)

        # Step 5: Mark as completed
        job.status       = "completed"
        job.results_path = str(results_path)
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
        job.status        = "failed"
        job.error_message = str(e)
        db.commit()

    finally:
        db.close()