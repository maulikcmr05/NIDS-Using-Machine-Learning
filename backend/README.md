# NIDS Backend API

FastAPI backend for the NIDS project.

## Setup

### 1. Go into backend folder
```
cd backend
```

### 2. Create virtual environment
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install libraries
```
pip install -r requirements.txt
```

### 4. Get model file from Person 2
Place `best_nids_model.pkl` inside `ml_artifacts/` folder.

### 5. Run the server
```
uvicorn main:app --reload
```

### 6. Open in browser
```
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload CSV file |
| GET | `/results/{job_id}/status` | Check status |
| GET | `/results/{job_id}` | Get predictions |
| GET | `/results/{job_id}/stats` | Get attack summary |
| GET | `/history` | List past uploads |