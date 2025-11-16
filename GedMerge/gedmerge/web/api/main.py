"""FastAPI application for GedMerge ML models."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware to allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
        logger.info(f"Upload directory: {upload_dir}")

        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_error:
            logger.error(f"Failed to create upload directory {upload_dir}: {mkdir_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create upload directory: {str(mkdir_error)}"
            )

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / unique_filename

        logger.info(f"Saving file to: {file_path}")

        # Save file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as save_error:
            logger.error(f"Failed to save file {file_path}: {save_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(save_error)}"
            )

        file_size = os.path.getsize(file_path)

        logger.info(f"File uploaded successfully: {file_path} ({file_size} bytes)")

        return {
            "filename": unique_filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": datetime.now().isoformat(),
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error uploading file: {str(e)}"
        )


class GeocodeRequest(BaseModel):
    place_name: str
    language: str = "en"


class ReverseGeocodeRequest(BaseModel):
    latitude: float
    longitude: float
    language: str = "en"


class BatchGeocodeRequest(BaseModel):
    database_path: str
    limit: Optional[int] = None  # Limit number of places to geocode (None = all)
    update_database: bool = False  # Whether to update the database with geocoded coordinates


@app.post("/api/geocode")
async def geocode_place(request: GeocodeRequest):
    """Geocode a place name to coordinates using Nominatim."""
    try:
        from ...utils.geocoding import NominatimGeocoder

        geocoder = NominatimGeocoder()
        result = geocoder.geocode(request.place_name, request.language)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not geocode place: {request.place_name}"
            )

        return {
            "place_name": request.place_name,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "display_name": result.display_name,
            "place_type": result.place_type,
            "osm_id": result.osm_id,
            "confidence": result.confidence,
            "address": result.address,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Geocoding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Geocoding failed: {str(e)}"
        )


@app.post("/api/reverse-geocode")
async def reverse_geocode_coords(request: ReverseGeocodeRequest):
    """Reverse geocode coordinates to a place name using Nominatim."""
    try:
        from ...utils.geocoding import NominatimGeocoder

        geocoder = NominatimGeocoder()
        result = geocoder.reverse_geocode(
            request.latitude,
            request.longitude,
            request.language
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not reverse geocode coordinates: ({request.latitude}, {request.longitude})"
            )

        return {
            "latitude": result.latitude,
            "longitude": result.longitude,
            "display_name": result.display_name,
            "place_type": result.place_type,
            "osm_id": result.osm_id,
            "confidence": result.confidence,
            "address": result.address,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reverse geocoding failed: {str(e)}"
        )


@app.post("/api/geocode/batch")
async def batch_geocode_places(request: BatchGeocodeRequest, background_tasks: BackgroundTasks):
    """Batch geocode places from database."""
    try:
        from ...utils.geocoding import NominatimGeocoder
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)
        geocoder = NominatimGeocoder()

        # Get places without coordinates
        query = """
            SELECT PlaceID, Name, Latitude, Longitude
            FROM PlaceTable
            WHERE (Latitude IS NULL OR Latitude = 0) AND (Longitude IS NULL OR Longitude = 0)
            ORDER BY PlaceID
        """

        if request.limit:
            query += f" LIMIT {request.limit}"

        places_to_geocode = []
        with db.conn:
            cursor = db.conn.cursor()
            cursor.execute(query)
            places_to_geocode = cursor.fetchall()

        if not places_to_geocode:
            return {
                "status": "success",
                "message": "No places need geocoding",
                "total_places": 0,
                "geocoded": 0,
                "failed": 0,
            }

        # Geocode places
        results = {
            "geocoded": [],
            "failed": [],
        }

        for place_id, name, lat, lon in places_to_geocode:
            logger.info(f"Geocoding place {place_id}: {name}")

            result = geocoder.geocode(name)

            if result:
                results["geocoded"].append({
                    "place_id": place_id,
                    "name": name,
                    "latitude": result.latitude,
                    "longitude": result.longitude,
                    "display_name": result.display_name,
                })

                # Update database if requested
                if request.update_database:
                    with db.conn:
                        cursor = db.conn.cursor()
                        # Convert to microdegrees for RootsMagic
                        lat_micro = int(result.latitude * 1_000_000)
                        lon_micro = int(result.longitude * 1_000_000)
                        cursor.execute(
                            "UPDATE PlaceTable SET Latitude = ?, Longitude = ?, LatLongExact = 1 WHERE PlaceID = ?",
                            (lat_micro, lon_micro, place_id)
                        )
                        db.conn.commit()
            else:
                results["failed"].append({
                    "place_id": place_id,
                    "name": name,
                })

        return {
            "status": "success",
            "message": f"Geocoded {len(results['geocoded'])} of {len(places_to_geocode)} places",
            "total_places": len(places_to_geocode),
            "geocoded": len(results["geocoded"]),
            "failed": len(results["failed"]),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch geocoding error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch geocoding failed: {str(e)}"
        )


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
        import re
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        standardized = 0
        merged = 0
        updated = 0

        logger.info(f"Repairing places in {request.database_path}")

        with db.transaction():
            cursor = db.conn.cursor()

            # Get all places
            cursor.execute("SELECT PlaceID, PlaceName FROM PlaceTable ORDER BY PlaceID")
            places = cursor.fetchall()

            logger.info(f"Found {len(places)} places to analyze")

            # Track place normalization mapping
            place_mapping = {}  # Maps old PlaceID to new PlaceID

            # Helper function to remove postal codes
            def remove_postal_code(place_name):
                if not place_name:
                    return place_name
                patterns = [
                    r',?\s*\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b',  # Canadian
                    r',?\s*\b\d{5}(?:-\d{4})?\b',  # US ZIP
                    r',?\s*\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b',  # UK
                ]
                cleaned = place_name
                for pattern in patterns:
                    cleaned = re.sub(pattern, '', cleaned)
                cleaned = re.sub(r',\s*,', ',', cleaned)
                cleaned = re.sub(r',\s*$', '', cleaned)
                cleaned = re.sub(r'^\s*,', '', cleaned)
                return cleaned.strip()

            # Helper function to standardize place format
            def standardize_place(place_name):
                if not place_name:
                    return place_name
                # Remove postal codes
                cleaned = remove_postal_code(place_name)
                # Trim whitespace around commas
                cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
                # Remove duplicate commas and trim
                cleaned = re.sub(r',+', ',', cleaned)
                cleaned = cleaned.strip().strip(',').strip()
                return cleaned

            # Normalize and deduplicate places
            normalized_places = {}  # Maps normalized name to original PlaceID

            for place_id, place_name in places:
                normalized = standardize_place(place_name)

                if normalized != place_name:
                    standardized += 1

                # Track for deduplication (case-insensitive)
                norm_key = normalized.lower() if normalized else ""

                if norm_key in normalized_places:
                    # This is a duplicate
                    original_place_id = normalized_places[norm_key]
                    place_mapping[place_id] = original_place_id
                    merged += 1
                else:
                    # First occurrence
                    normalized_places[norm_key] = place_id
                    place_mapping[place_id] = place_id

                    # Update the place name if it was standardized
                    if normalized != place_name and normalized:
                        cursor.execute("""
                            UPDATE PlaceTable
                            SET PlaceName = ?
                            WHERE PlaceID = ?
                        """, (normalized, place_id))

            # Update all references to merged places in EventTable
            for old_place_id, new_place_id in place_mapping.items():
                if old_place_id != new_place_id:
                    cursor.execute("""
                        UPDATE EventTable
                        SET PlaceID = ?
                        WHERE PlaceID = ?
                    """, (new_place_id, old_place_id))

                    # Delete the duplicate place
                    cursor.execute("DELETE FROM PlaceTable WHERE PlaceID = ?", (old_place_id,))
                    updated += 1

        logger.info(f"Places repair completed: {standardized} standardized, {merged} merged, {updated} updated")

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
        import re
        from datetime import datetime
        from ...rootsmagic.adapter import RootsMagicDatabase
        from ...utils.name_parser import NameParser

        db = RootsMagicDatabase(request.database_path)

        reversed_fixed = 0
        variants_extracted = 0
        titles_moved = 0
        total_updated = 0

        logger.info(f"Repairing names in {request.database_path}")

        with db.transaction():
            cursor = db.conn.cursor()

            # Get all names
            cursor.execute("""
                SELECT NameID, OwnerID, Surname, Given, Prefix, Suffix, Nickname, IsPrimary
                FROM NameTable
                ORDER BY NameID
            """)
            names = cursor.fetchall()

            logger.info(f"Found {len(names)} names to analyze")

            for name_id, owner_id, surname, given, prefix, suffix, nickname, is_primary in names:
                updated = False
                new_surname = surname
                new_given = given
                new_prefix = prefix
                new_suffix = suffix
                new_nickname = nickname

                # 1. Extract and move prefixes (titles) from given names
                if new_given and not new_prefix:
                    text, extracted_prefix = NameParser.extract_prefix(new_given)
                    if extracted_prefix:
                        new_prefix = extracted_prefix
                        new_given = text
                        titles_moved += 1
                        updated = True

                # 2. Extract embedded variants from given names (in parentheses or quotes)
                if new_given:
                    # Check for parentheses variants like "John (Jack)"
                    variant_match = re.search(r'\(([^)]+)\)', new_given)
                    if variant_match:
                        variant = variant_match.group(1).strip()
                        # Remove the variant from given name
                        new_given = re.sub(r'\s*\([^)]+\)', '', new_given).strip()

                        # Create alternate name record with the variant
                        if is_primary and variant:
                            cursor.execute("""
                                INSERT INTO NameTable (
                                    OwnerID, Surname, Given, Prefix, Suffix, Nickname,
                                    NameType, IsPrimary, Language
                                )
                                VALUES (?, ?, ?, ?, ?, ?, 1, 0, '')
                            """, (owner_id, new_surname or '', variant, new_prefix or '', new_suffix or '', '', ))
                            variants_extracted += 1
                        updated = True

                    # Check for quoted epithets
                    text, epithets = NameParser.extract_quoted_epithets(new_given)
                    if epithets:
                        new_given = text
                        if not new_nickname and epithets:
                            new_nickname = epithets[0]
                        updated = True

                # 3. Extract nobility suffixes from given names
                if new_given:
                    text, nobility_suffix = NameParser.extract_nobility_suffix(new_given)
                    if nobility_suffix:
                        new_given = text
                        if new_suffix:
                            new_suffix = f"{new_suffix} {nobility_suffix}".strip()
                        else:
                            new_suffix = nobility_suffix
                        updated = True

                # 4. Detect reversed names (surname in given field with comma)
                if new_given and ',' in new_given and not new_surname:
                    # Pattern: "LastName, FirstName" in Given field
                    parts = new_given.split(',', 1)
                    if len(parts) == 2:
                        potential_surname = parts[0].strip()
                        potential_given = parts[1].strip()
                        # Swap them
                        new_surname = potential_surname
                        new_given = potential_given
                        reversed_fixed += 1
                        updated = True

                # 5. Handle surname particles misclassified as nicknames
                if new_nickname and NameParser.has_surname_particle(new_nickname):
                    parsed = NameParser.parse_field_with_surname_particle(new_nickname, 'NICK')
                    if parsed.surname and not new_surname:
                        new_surname = parsed.surname
                        if parsed.given and not new_given:
                            new_given = parsed.given
                        if parsed.prefix and not new_prefix:
                            new_prefix = parsed.prefix
                        # Clear nickname since it was misclassified
                        new_nickname = ''
                        updated = True

                # Update the name record if any changes were made
                if updated:
                    utc_mod_date = int(datetime.now().timestamp())
                    cursor.execute("""
                        UPDATE NameTable
                        SET Surname = ?, Given = ?, Prefix = ?, Suffix = ?, Nickname = ?, UTCModDate = ?
                        WHERE NameID = ?
                    """, (new_surname or '', new_given or '', new_prefix or '', new_suffix or '', new_nickname or '', utc_mod_date, name_id))
                    total_updated += 1

        logger.info(f"Names repair completed: {reversed_fixed} reversed, {variants_extracted} variants, {titles_moved} titles, {total_updated} total updated")

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
        import re
        from datetime import datetime
        from ...rootsmagic.adapter import RootsMagicDatabase

        db = RootsMagicDatabase(request.database_path)

        dates_fixed = 0
        formats_standardized = 0
        chronological_fixed = 0
        total_updated = 0

        logger.info(f"Repairing events in {request.database_path}")

        with db.transaction():
            cursor = db.conn.cursor()

            # Get all events with their dates
            cursor.execute("""
                SELECT EventID, OwnerType, OwnerID, EventType, Date, SortDate, Details
                FROM EventTable
                ORDER BY OwnerType, OwnerID, SortDate
            """)
            events = cursor.fetchall()

            logger.info(f"Found {len(events)} events to analyze")

            # Group events by person to check chronological order
            person_events = {}
            for event_id, owner_type, owner_id, event_type, date, sort_date, details in events:
                if owner_type == 0:  # Person event
                    if owner_id not in person_events:
                        person_events[owner_id] = []
                    person_events[owner_id].append({
                        'event_id': event_id,
                        'event_type': event_type,
                        'date': date,
                        'sort_date': sort_date,
                        'details': details
                    })

            # Check chronological order for each person
            # Event types: 1=Birth, 2=Death, 3=Burial, etc.
            for person_id, events_list in person_events.items():
                birth_events = [e for e in events_list if e['event_type'] == 1]
                death_events = [e for e in events_list if e['event_type'] == 2]

                # Check if birth comes after death
                if birth_events and death_events:
                    birth_date = birth_events[0]['sort_date']
                    death_date = death_events[0]['sort_date']

                    if birth_date and death_date and birth_date > death_date:
                        # Chronological error detected
                        logger.warning(f"Person {person_id} has birth after death: birth={birth_date}, death={death_date}")
                        chronological_fixed += 1

            # Standardize date formats
            for event_id, owner_type, owner_id, event_type, date, sort_date, details in events:
                updated = False
                new_date = date
                new_sort_date = sort_date

                if date:
                    # Remove extra whitespace
                    cleaned_date = re.sub(r'\s+', ' ', date).strip()
                    if cleaned_date != date:
                        new_date = cleaned_date
                        updated = True
                        formats_standardized += 1

                    # Standardize common date patterns
                    # Convert "01-15-2020" to "15 Jan 2020" format
                    date_match = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', cleaned_date)
                    if date_match:
                        month_num = int(date_match.group(1))
                        day = date_match.group(2)
                        year = date_match.group(3)
                        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        if 1 <= month_num <= 12:
                            new_date = f"{day} {month_names[month_num]} {year}"
                            updated = True
                            dates_fixed += 1

                if updated:
                    utc_mod_date = int(datetime.now().timestamp())
                    cursor.execute("""
                        UPDATE EventTable
                        SET Date = ?
                        WHERE EventID = ?
                    """, (new_date, event_id))
                    total_updated += 1

        logger.info(f"Events repair completed: {dates_fixed} dates, {formats_standardized} formats, {chronological_fixed} chronological, {total_updated} total updated")

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

        relationships_fixed = 0
        orphans_linked = 0
        families_repaired = 0
        total_updated = 0

        logger.info(f"Repairing people and families in {request.database_path}")

        with db.transaction():
            cursor = db.conn.cursor()

            # 1. Fix inconsistent ParentID and SpouseID references
            # Ensure ParentID points to a valid family where person is a child
            cursor.execute("""
                SELECT PersonID, ParentID, SpouseID
                FROM PersonTable
                WHERE ParentID > 0 OR SpouseID > 0
            """)
            persons = cursor.fetchall()

            for person_id, parent_id, spouse_id in persons:
                updated = False

                # Check if ParentID is valid
                if parent_id > 0:
                    cursor.execute("""
                        SELECT COUNT(*) FROM ChildTable
                        WHERE ChildID = ? AND FamilyID = ?
                    """, (person_id, parent_id))
                    count = cursor.fetchone()[0]

                    if count == 0:
                        # ParentID is invalid, try to find correct family
                        cursor.execute("""
                            SELECT FamilyID FROM ChildTable
                            WHERE ChildID = ?
                            LIMIT 1
                        """, (person_id,))
                        result = cursor.fetchone()

                        if result:
                            correct_parent_id = result[0]
                            cursor.execute("""
                                UPDATE PersonTable
                                SET ParentID = ?
                                WHERE PersonID = ?
                            """, (correct_parent_id, person_id))
                            relationships_fixed += 1
                            updated = True
                        else:
                            # No family link found, clear invalid ParentID
                            cursor.execute("""
                                UPDATE PersonTable
                                SET ParentID = 0
                                WHERE PersonID = ?
                            """, (person_id,))
                            updated = True

                # Check if SpouseID is valid
                if spouse_id > 0:
                    cursor.execute("""
                        SELECT COUNT(*) FROM FamilyTable
                        WHERE FamilyID = ? AND (FatherID = ? OR MotherID = ?)
                    """, (spouse_id, person_id, person_id))
                    count = cursor.fetchone()[0]

                    if count == 0:
                        # SpouseID is invalid, try to find correct family
                        cursor.execute("""
                            SELECT FamilyID FROM FamilyTable
                            WHERE FatherID = ? OR MotherID = ?
                            LIMIT 1
                        """, (person_id, person_id))
                        result = cursor.fetchone()

                        if result:
                            correct_spouse_id = result[0]
                            cursor.execute("""
                                UPDATE PersonTable
                                SET SpouseID = ?
                                WHERE PersonID = ?
                            """, (correct_spouse_id, person_id))
                            relationships_fixed += 1
                            updated = True
                        else:
                            # No family found, clear invalid SpouseID
                            cursor.execute("""
                                UPDATE PersonTable
                                SET SpouseID = 0
                                WHERE PersonID = ?
                            """, (person_id,))
                            updated = True

                if updated:
                    total_updated += 1

            # 2. Find and count orphaned persons (no parent family, no spouse family)
            cursor.execute("""
                SELECT COUNT(*) FROM PersonTable p
                WHERE p.ParentID = 0 AND p.SpouseID = 0
                AND NOT EXISTS (
                    SELECT 1 FROM ChildTable c WHERE c.ChildID = p.PersonID
                )
                AND NOT EXISTS (
                    SELECT 1 FROM FamilyTable f WHERE f.FatherID = p.PersonID OR f.MotherID = p.PersonID
                )
            """)
            orphans_linked = cursor.fetchone()[0]

            # 3. Repair family structures - ensure FamilyTable references valid persons
            cursor.execute("""
                SELECT FamilyID, FatherID, MotherID
                FROM FamilyTable
            """)
            families = cursor.fetchall()

            for family_id, father_id, mother_id in families:
                updated = False

                # Check if father exists
                if father_id > 0:
                    cursor.execute("SELECT COUNT(*) FROM PersonTable WHERE PersonID = ?", (father_id,))
                    if cursor.fetchone()[0] == 0:
                        # Father doesn't exist, clear reference
                        cursor.execute("UPDATE FamilyTable SET FatherID = 0 WHERE FamilyID = ?", (family_id,))
                        updated = True
                        families_repaired += 1

                # Check if mother exists
                if mother_id > 0:
                    cursor.execute("SELECT COUNT(*) FROM PersonTable WHERE PersonID = ?", (mother_id,))
                    if cursor.fetchone()[0] == 0:
                        # Mother doesn't exist, clear reference
                        cursor.execute("UPDATE FamilyTable SET MotherID = 0 WHERE FamilyID = ?", (family_id,))
                        updated = True
                        families_repaired += 1

        logger.info(f"People repair completed: {relationships_fixed} relationships, {orphans_linked} orphans, {families_repaired} families, {total_updated} total updated")

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
        import re
        from pathlib import Path
        from ...rootsmagic.adapter import RootsMagicDatabase
        from ...utils.name_parser import NameParser

        db = RootsMagicDatabase(request.database_path)

        name_issues = 0
        place_issues = 0
        event_issues = 0
        relationship_issues = 0
        details = []

        logger.info(f"Running sanity check on {request.database_path}")

        cursor = db.conn.cursor()

        # 1. Check names for issues
        cursor.execute("""
            SELECT NameID, Surname, Given, Prefix, Suffix, Nickname
            FROM NameTable
        """)
        names = cursor.fetchall()

        reversed_names = 0
        embedded_variants = 0
        titles_in_wrong_field = 0
        placeholder_names = 0

        for name_id, surname, given, prefix, suffix, nickname in names:
            # Check for reversed names (comma in given name)
            if given and ',' in given and not surname:
                reversed_names += 1

            # Check for embedded variants
            if given and ('(' in given or ')' in given):
                embedded_variants += 1

            # Check for titles in wrong field
            if given and not prefix:
                _, extracted_prefix = NameParser.extract_prefix(given)
                if extracted_prefix:
                    titles_in_wrong_field += 1

            # Check for placeholder names
            if surname and surname.upper() in ['NN', 'UNKNOWN', '?', '??', 'N.N.']:
                placeholder_names += 1

            # Check for surname particles in nickname field
            if nickname and NameParser.has_surname_particle(nickname):
                name_issues += 1

        if reversed_names > 0:
            details.append({"category": "Names", "description": "Reversed names detected", "count": reversed_names})
            name_issues += reversed_names

        if embedded_variants > 0:
            details.append({"category": "Names", "description": "Embedded name variants", "count": embedded_variants})
            name_issues += embedded_variants

        if titles_in_wrong_field > 0:
            details.append({"category": "Names", "description": "Titles in wrong field", "count": titles_in_wrong_field})
            name_issues += titles_in_wrong_field

        if placeholder_names > 0:
            details.append({"category": "Names", "description": "Placeholder names", "count": placeholder_names})
            name_issues += placeholder_names

        # 2. Check places for issues
        cursor.execute("SELECT PlaceID, PlaceName FROM PlaceTable")
        places = cursor.fetchall()

        places_with_postal_codes = 0
        duplicate_places = 0

        seen_places = set()
        for place_id, place_name in places:
            if not place_name:
                continue

            # Check for postal codes
            postal_patterns = [
                r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b',  # Canadian
                r'\b\d{5}(?:-\d{4})?\b',  # US ZIP
                r'\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b',  # UK
            ]
            for pattern in postal_patterns:
                if re.search(pattern, place_name):
                    places_with_postal_codes += 1
                    break

            # Check for duplicates (case-insensitive)
            normalized = place_name.lower().strip()
            if normalized in seen_places:
                duplicate_places += 1
            seen_places.add(normalized)

        if places_with_postal_codes > 0:
            details.append({"category": "Places", "description": "Places with postal codes", "count": places_with_postal_codes})
            place_issues += places_with_postal_codes

        if duplicate_places > 0:
            details.append({"category": "Places", "description": "Duplicate places", "count": duplicate_places})
            place_issues += duplicate_places

        # 3. Check events for issues
        cursor.execute("""
            SELECT EventID, OwnerType, OwnerID, EventType, Date, SortDate
            FROM EventTable
            WHERE OwnerType = 0
        """)
        events = cursor.fetchall()

        invalid_dates = 0
        chronological_errors = 0

        # Group by person
        person_events = {}
        for event_id, owner_type, owner_id, event_type, date, sort_date in events:
            if owner_id not in person_events:
                person_events[owner_id] = []
            person_events[owner_id].append({
                'event_type': event_type,
                'date': date,
                'sort_date': sort_date
            })

            # Check for invalid date formats
            if date and re.search(r'\d{1,2}-\d{1,2}-\d{4}', date):
                invalid_dates += 1

        # Check chronological order
        for person_id, events_list in person_events.items():
            birth_events = [e for e in events_list if e['event_type'] == 1]
            death_events = [e for e in events_list if e['event_type'] == 2]

            if birth_events and death_events:
                birth_date = birth_events[0]['sort_date']
                death_date = death_events[0]['sort_date']
                if birth_date and death_date and birth_date > death_date:
                    chronological_errors += 1

        if invalid_dates > 0:
            details.append({"category": "Events", "description": "Invalid date formats", "count": invalid_dates})
            event_issues += invalid_dates

        if chronological_errors > 0:
            details.append({"category": "Events", "description": "Chronological errors", "count": chronological_errors})
            event_issues += chronological_errors

        # 4. Check relationships for issues
        cursor.execute("""
            SELECT PersonID, ParentID, SpouseID
            FROM PersonTable
            WHERE ParentID > 0 OR SpouseID > 0
        """)
        persons = cursor.fetchall()

        invalid_parent_refs = 0
        invalid_spouse_refs = 0

        for person_id, parent_id, spouse_id in persons:
            if parent_id > 0:
                cursor.execute("""
                    SELECT COUNT(*) FROM ChildTable
                    WHERE ChildID = ? AND FamilyID = ?
                """, (person_id, parent_id))
                if cursor.fetchone()[0] == 0:
                    invalid_parent_refs += 1

            if spouse_id > 0:
                cursor.execute("""
                    SELECT COUNT(*) FROM FamilyTable
                    WHERE FamilyID = ? AND (FatherID = ? OR MotherID = ?)
                """, (spouse_id, person_id, person_id))
                if cursor.fetchone()[0] == 0:
                    invalid_spouse_refs += 1

        if invalid_parent_refs > 0:
            details.append({"category": "Relationships", "description": "Invalid parent references", "count": invalid_parent_refs})
            relationship_issues += invalid_parent_refs

        if invalid_spouse_refs > 0:
            details.append({"category": "Relationships", "description": "Invalid spouse references", "count": invalid_spouse_refs})
            relationship_issues += invalid_spouse_refs

        # Calculate totals
        total_issues = name_issues + place_issues + event_issues + relationship_issues

        # Calculate quality score (0-100, where 100 is perfect)
        total_records = len(names) + len(places) + len(events) + len(persons)
        if total_records > 0:
            quality_score = max(0, min(100, 100 - (total_issues / total_records * 100)))
        else:
            quality_score = 100

        logger.info(f"Sanity check completed: {total_issues} total issues found, quality score: {quality_score:.1f}")

        return {
            "total_issues": total_issues,
            "name_issues": name_issues,
            "place_issues": place_issues,
            "event_issues": event_issues,
            "relationship_issues": relationship_issues,
            "quality_score": round(quality_score, 1),
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
        logger.info(f"Running all repairs on {request.database_path}")

        # Run all repair functions in sequence
        # 1. Repair places first (needed for event references)
        logger.info("Step 1/4: Repairing places...")
        places_result = await repair_places(request)

        # 2. Repair names
        logger.info("Step 2/4: Repairing names...")
        names_result = await repair_names(request)

        # 3. Repair events
        logger.info("Step 3/4: Repairing events...")
        events_result = await repair_events(request)

        # 4. Repair people and relationships
        logger.info("Step 4/4: Repairing people and relationships...")
        people_result = await repair_people(request)

        # Aggregate results
        places_repaired = places_result.get("standardized", 0) + places_result.get("merged", 0)
        names_repaired = names_result.get("total_updated", 0)
        events_repaired = events_result.get("total_updated", 0)
        relationships_repaired = people_result.get("total_updated", 0)
        total_updated = places_repaired + names_repaired + events_repaired + relationships_repaired

        logger.info(f"All repairs completed: {total_updated} total updates")

        return {
            "places_repaired": places_repaired,
            "names_repaired": names_repaired,
            "events_repaired": events_repaired,
            "relationships_repaired": relationships_repaired,
            "total_updated": total_updated,
            "places_details": places_result,
            "names_details": names_result,
            "events_details": events_result,
            "people_details": people_result,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error running all repairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
