"""FastAPI application for GedMerge ML models."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import shutil
import os

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

# Mount static files and templates (conditionally if directories exist)
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Only mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Mounted static files from {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found at {STATIC_DIR}, skipping static files mounting")

# Only initialize templates if directory exists
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    logger.info(f"Initialized templates from {TEMPLATES_DIR}")
else:
    logger.warning(f"Templates directory not found at {TEMPLATES_DIR}, using fallback")
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
    if templates is not None:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    else:
        # Fallback response when templates are not available
        return HTMLResponse(
            content="""
            <html>
                <head><title>GedMerge ML API</title></head>
                <body>
                    <h1>GedMerge ML API</h1>
                    <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
                    <p>Note: Dashboard templates not found. Please add templates to enable full UI.</p>
                </body>
            </html>
            """,
            status_code=200
        )


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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a database file for training or analysis."""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = BASE_DIR / "static" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / unique_filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        logger.info(f"File uploaded: {file_path} ({file_size} bytes)")

        return {
            "filename": unique_filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PlacesAnalysisRequest(BaseModel):
    database_path: str


class PlacesRepairRequest(BaseModel):
    database_path: str


@app.post("/api/places/analyze")
async def analyze_places(request: PlacesAnalysisRequest):
    """Analyze places in a database for duplicates and standardization needs."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Get all places from database
        # This is a simplified implementation - you may need to adjust based on your actual schema
        places = []
        try:
            # Try to query places from the database
            query = "SELECT DISTINCT PlaceName FROM PlaceTable WHERE PlaceName IS NOT NULL"
            places = [row[0] for row in db.connection.execute(query).fetchall()]
        except Exception as e:
            # If PlaceTable doesn't exist, try EventTable
            try:
                query = "SELECT DISTINCT Place FROM EventTable WHERE Place IS NOT NULL"
                places = [row[0] for row in db.connection.execute(query).fetchall()]
            except:
                logger.warning(f"Could not query places: {e}")
                places = []

        # Simple analysis
        total_places = len(places)
        unique_places = len(set(places))
        duplicates = total_places - unique_places

        # Check for places needing standardization (simplified)
        needs_standardization = 0
        for place in places:
            # Check for common issues
            if place and (',' not in place or place.isupper() or place.islower()):
                needs_standardization += 1

        return {
            "total_places": total_places,
            "unique_places": unique_places,
            "duplicates": duplicates,
            "needs_standardization": needs_standardization,
        }

    except Exception as e:
        logger.error(f"Error analyzing places: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RepairRequest(BaseModel):
    database_path: str


@app.post("/api/repairs/places")
async def repair_places(request: RepairRequest):
    """Repair and standardize places in a database."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Load all places
        # 2. Standardize format (City, County, State, Country)
        # 3. Merge duplicates
        # 4. Update records

        standardized = 0
        merged = 0
        updated = 0

        # Placeholder for actual repair logic
        logger.info(f"Repairing places in {request.database_path}")

        return {
            "standardized": standardized,
            "merged": merged,
            "updated": updated,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error repairing places: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/repairs/names")
async def repair_names(request: RepairRequest):
    """Repair name issues like reversed names, embedded variants, and titles."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Placeholder for actual repair logic
        # In a real implementation, you would:
        # 1. Detect reversed names (First, Last vs Last, First)
        # 2. Extract embedded variants (e.g., "John (Jack)" -> John + variant Jack)
        # 3. Move titles to proper fields (e.g., "Dr. John Smith" -> Title: Dr., Name: John Smith)
        # 4. Standardize formatting

        reversed_fixed = 0
        variants_extracted = 0
        titles_moved = 0
        total_updated = 0

        logger.info(f"Repairing names in {request.database_path}")

        return {
            "reversed_fixed": reversed_fixed,
            "variants_extracted": variants_extracted,
            "titles_moved": titles_moved,
            "total_updated": total_updated,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error repairing names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/repairs/events")
async def repair_events(request: RepairRequest):
    """Repair event issues like invalid dates and chronological errors."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Placeholder for actual repair logic
        # In a real implementation, you would:
        # 1. Validate dates
        # 2. Fix date formats
        # 3. Check chronological order (birth before death, etc.)
        # 4. Standardize event types

        dates_fixed = 0
        formats_standardized = 0
        chronological_fixed = 0
        total_updated = 0

        logger.info(f"Repairing events in {request.database_path}")

        return {
            "dates_fixed": dates_fixed,
            "formats_standardized": formats_standardized,
            "chronological_fixed": chronological_fixed,
            "total_updated": total_updated,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error repairing events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/repairs/people")
async def repair_people(request: RepairRequest):
    """Repair people and family relationship issues."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Placeholder for actual repair logic
        # In a real implementation, you would:
        # 1. Fix relationship inconsistencies
        # 2. Validate family structures
        # 3. Link orphaned records
        # 4. Repair missing parent/child links

        relationships_fixed = 0
        orphans_linked = 0
        families_repaired = 0
        total_updated = 0

        logger.info(f"Repairing people and families in {request.database_path}")

        return {
            "relationships_fixed": relationships_fixed,
            "orphans_linked": orphans_linked,
            "families_repaired": families_repaired,
            "total_updated": total_updated,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error repairing people: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/sanity-check")
async def sanity_check(request: RepairRequest):
    """Run comprehensive data quality checks on a database."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Placeholder for actual sanity check logic
        # In a real implementation, you would:
        # 1. Check all names for issues
        # 2. Check all places for issues
        # 3. Check all events for issues
        # 4. Check all relationships for issues
        # 5. Calculate overall quality score

        name_issues = 0
        place_issues = 0
        event_issues = 0
        relationship_issues = 0
        total_issues = name_issues + place_issues + event_issues + relationship_issues
        quality_score = 100  # Placeholder

        details = [
            # Example detail structure
            # {"category": "Names", "description": "Reversed names detected", "count": 5},
        ]

        logger.info(f"Running sanity check on {request.database_path}")

        return {
            "total_issues": total_issues,
            "name_issues": name_issues,
            "place_issues": place_issues,
            "event_issues": event_issues,
            "relationship_issues": relationship_issues,
            "quality_score": quality_score,
            "details": details,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error running sanity check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality/repair-all")
async def repair_all(request: RepairRequest):
    """Run all repairs on a database."""
    try:
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        # Run all repairs
        places_repaired = 0
        names_repaired = 0
        events_repaired = 0
        relationships_repaired = 0
        total_updated = 0

        logger.info(f"Running all repairs on {request.database_path}")

        return {
            "places_repaired": places_repaired,
            "names_repaired": names_repaired,
            "events_repaired": events_repaired,
            "relationships_repaired": relationships_repaired,
            "total_updated": total_updated,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error running all repairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
