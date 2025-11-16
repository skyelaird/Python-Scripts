"""FastAPI application for GedMerge ML models."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from ...ml.models import (
    DuplicateDetectionModel,
    NameMatchingModel,
    LanguageDetectionModel,
    RelationshipInferenceModel,
    DataQualityClassifier,
)
from ...ml.utils import ModelRegistry, MLConfig
from ...rootsmagic.adapter import RootsMagicDatabase

# Import continual learning router
from .continual_learning import router as learning_router

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GedMerge ML API",
    description="Machine Learning API for genealogy duplicate detection and data quality with continual learning",
    version="2.0.0",
)

# Include continual learning router
app.include_router(learning_router)

# Mount static files and templates
BASE_DIR = Path(__file__).parent.parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

# Only mount static files if directory exists
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Only initialize templates if directory exists
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    logger.warning(f"Templates directory not found: {templates_dir}")
    templates = None

# Global model registry
config = MLConfig()
registry = ModelRegistry(config.model_dir)

# Model instances (lazy loaded)
models = {
    'duplicate_detector': None,
    'name_matcher': None,
    'language_detector': None,
    'relationship_gnn': None,
    'quality_classifier': None,
}

# Training job queue
training_jobs = {}


# Pydantic models for API
class NamePair(BaseModel):
    name1: str
    name2: str


class DuplicateCheckRequest(BaseModel):
    person1_id: str
    person2_id: str
    database_path: str


class LanguageDetectionRequest(BaseModel):
    name: str


class QualityCheckRequest(BaseModel):
    person_id: str
    database_path: str


class TrainingJobRequest(BaseModel):
    model_type: str
    database_path: str
    config: Optional[Dict[str, Any]] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    model_type: str
    status: str
    created_at: str


# Helper functions
def load_model(model_type: str):
    """Load a model from registry."""
    if models[model_type] is None:
        try:
            if model_type == 'duplicate_detector':
                model, _ = registry.load_model('duplicate_detector')
                models[model_type] = model
            elif model_type == 'name_matcher':
                model, _ = registry.load_model('name_matcher')
                models[model_type] = model
            elif model_type == 'language_detector':
                model, _ = registry.load_model('language_detector')
                models[model_type] = model
            elif model_type == 'relationship_gnn':
                model, _ = registry.load_model('relationship_gnn')
                models[model_type] = model
            elif model_type == 'quality_classifier':
                model, _ = registry.load_model('quality_classifier')
                models[model_type] = model
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            logger.info(f"Loaded {model_type} model")
        except Exception as e:
            logger.warning(f"Could not load {model_type}: {e}")
            return None

    return models[model_type]


# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard page."""
    if templates is None:
        return HTMLResponse(
            content="<h1>GedMerge ML API</h1><p>API is running. Visit <a href='/docs'>/docs</a> for API documentation.</p>",
            status_code=200
        )
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/models")
async def list_models():
    """List all registered models."""
    models_list = registry.list_models()

    model_info = {}
    for model_name, versions in models_list.items():
        latest_version = registry.get_latest_version(model_name)
        info = registry.get_model_info(model_name, latest_version)
        model_info[model_name] = {
            'versions': versions,
            'latest_version': latest_version,
            'metrics': info.get('metrics', {}),
            'registered_at': info.get('registered_at'),
        }

    return model_info


@app.post("/api/predict/duplicate")
async def predict_duplicate(request: DuplicateCheckRequest):
    """Check if two persons are duplicates."""
    model = load_model('duplicate_detector')

    if model is None:
        raise HTTPException(status_code=503, detail="Duplicate detector model not available")

    try:
        # Load database
        db = RootsMagicDatabase(request.database_path)

        # Get persons
        person1 = db.get_person(request.person1_id)
        person2 = db.get_person(request.person2_id)

        if not person1 or not person2:
            raise HTTPException(status_code=404, detail="Person not found")

        # Predict
        prediction = model.predict(person1, person2)

        return {
            'is_duplicate': prediction.is_duplicate,
            'confidence': prediction.confidence,
            'feature_importances': prediction.feature_importances,
        }

    except Exception as e:
        logger.error(f"Error in duplicate prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/name-match")
async def predict_name_match(request: NamePair):
    """Predict similarity between two names."""
    model = load_model('name_matcher')

    if model is None:
        raise HTTPException(status_code=503, detail="Name matcher model not available")

    try:
        result = model.predict(request.name1, request.name2)

        return {
            'name1': result.name1,
            'name2': result.name2,
            'similarity': result.similarity,
            'is_match': result.is_match,
        }

    except Exception as e:
        logger.error(f"Error in name matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/language")
async def predict_language(request: LanguageDetectionRequest):
    """Detect language of a name."""
    model = load_model('language_detector')

    if model is None:
        raise HTTPException(status_code=503, detail="Language detector model not available")

    try:
        language, confidence = model.predict(request.name)

        return {
            'name': request.name,
            'language': language,
            'confidence': confidence,
        }

    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/quality")
async def predict_quality(request: QualityCheckRequest):
    """Predict data quality issues for a person."""
    model = load_model('quality_classifier')

    if model is None:
        raise HTTPException(status_code=503, detail="Quality classifier model not available")

    try:
        # Load database
        db = RootsMagicDatabase(request.database_path)
        person = db.get_person(request.person_id)

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Predict
        predictions = model.predict(person)
        issues = model.get_quality_issues(person, threshold=0.5)

        return {
            'person_id': request.person_id,
            'predictions': predictions,
            'issues': issues,
        }

    except Exception as e:
        logger.error(f"Error in quality prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train", response_model=TrainingJobResponse)
async def start_training(request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Start a training job."""
    import uuid

    job_id = str(uuid.uuid4())

    training_jobs[job_id] = {
        'job_id': job_id,
        'model_type': request.model_type,
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'database_path': request.database_path,
        'config': request.config or {},
    }

    # Add to background tasks
    background_tasks.add_task(run_training_job, job_id, request)

    return TrainingJobResponse(**training_jobs[job_id])


@app.get("/api/train/{job_id}")
async def get_training_job(job_id: str):
    """Get training job status."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    return training_jobs[job_id]


@app.get("/api/train")
async def list_training_jobs():
    """List all training jobs."""
    return list(training_jobs.values())


async def run_training_job(job_id: str, request: TrainingJobRequest):
    """Run a training job in the background."""
    training_jobs[job_id]['status'] = 'running'
    training_jobs[job_id]['started_at'] = datetime.now().isoformat()

    try:
        # Import training module
        from ...ml.training import ModelTrainer

        trainer = ModelTrainer(config)

        # Train model
        if request.model_type == 'duplicate_detector':
            metrics = trainer.train_duplicate_detector(request.database_path)
        elif request.model_type == 'name_matcher':
            metrics = trainer.train_name_matcher(request.database_path)
        elif request.model_type == 'language_detector':
            metrics = trainer.train_language_detector(request.database_path)
        elif request.model_type == 'quality_classifier':
            metrics = trainer.train_quality_classifier(request.database_path)
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")

        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['metrics'] = metrics
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()

    except Exception as e:
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['failed_at'] = datetime.now().isoformat()
        logger.error(f"Training job {job_id} failed: {e}")


@app.get("/api/metrics/{model_name}")
async def get_model_metrics(model_name: str, version: Optional[str] = None):
    """Get metrics for a model."""
    try:
        info = registry.get_model_info(model_name, version)
        return {
            'model_name': model_name,
            'version': info.get('version'),
            'metrics': info.get('metrics', {}),
            'registered_at': info.get('registered_at'),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
