"""
GenAI Sepsis Classification API
Streamlined API for the GenAI-based sepsis early warning system.

Endpoints:
- POST /classify - Single patient classification
- POST /classify-batch - Batch classification
- GET /health - Health check
- GET /guardrail/thresholds - Simplified threshold values
- GET /guardrail/config - Full guardrail configuration (for SMEs)
- PUT /guardrail/config - Update guardrail configuration (for SMEs)
- POST /guardrail/reload - Hot-reload guardrail config from file
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from functools import wraps

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# API Configuration
# ============================================================================

app = FastAPI(
    title="GenAI Sepsis Classification API",
    description="AI-powered sepsis early warning system with 6-hour predictive risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key for authentication
API_KEY = os.getenv("API_KEY", "sepsis_api_key_2024")

# ============================================================================
# Request/Response Models
# ============================================================================

class PatientVitals(BaseModel):
    """Patient vital signs and lab values."""
    HR: Optional[float] = Field(None, description="Heart Rate (bpm)")
    SBP: Optional[float] = Field(None, description="Systolic Blood Pressure (mmHg)")
    DBP: Optional[float] = Field(None, description="Diastolic Blood Pressure (mmHg)")
    MAP: Optional[float] = Field(None, description="Mean Arterial Pressure (mmHg)")
    Temp: Optional[float] = Field(None, description="Temperature (Celsius)")
    Resp: Optional[float] = Field(None, description="Respiratory Rate (/min)")
    O2Sat: Optional[float] = Field(None, description="Oxygen Saturation (%)")
    WBC: Optional[float] = Field(None, description="White Blood Cell Count (K/µL)")
    Lactate: Optional[float] = Field(None, description="Lactate (mmol/L)")
    Creatinine: Optional[float] = Field(None, description="Creatinine (mg/dL)")
    Bilirubin_total: Optional[float] = Field(None, description="Total Bilirubin (mg/dL)")
    Platelets: Optional[float] = Field(None, description="Platelet Count (K/µL)")
    pH: Optional[float] = Field(None, description="Blood pH")
    Age: Optional[int] = Field(None, description="Patient Age")
    Gender: Optional[int] = Field(None, description="Gender (0=Female, 1=Male)")
    
    class Config:
        extra = "allow"  # Allow additional fields


class ClassifyRequest(BaseModel):
    """Request model for single patient classification."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    vitals: Dict[str, Any] = Field(..., description="Patient vital signs and lab values")
    notes: Optional[str] = Field(None, description="Clinician/nursing notes")


class BatchClassifyRequest(BaseModel):
    """Request model for batch classification."""
    patients: List[ClassifyRequest] = Field(..., description="List of patients to classify")


class ClassificationResponse(BaseModel):
    """Response model for classification."""
    request_id: str
    patient_id: Optional[str]
    status: str
    risk_score: int
    priority: str
    sepsis_probability_6h: str
    clinical_rationale: str
    alert_level: str
    alert_color: str
    action_required: str
    guardrail_override: bool
    override_reasons: List[str]
    total_processing_time_ms: float


# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for protected endpoints."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key


def require_auth(func):
    """Decorator to require API key authentication."""
    @wraps(func)
    async def wrapper(*args, x_api_key: str = Header(None), **kwargs):
        verify_api_key(x_api_key)
        return await func(*args, **kwargs)
    return wrapper


# ============================================================================
# GenAI Pipeline (Lazy Loading)
# ============================================================================

_pipeline = None

def get_pipeline():
    """Get or create the GenAI pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from genai_pipeline import GenAISepsisPipeline
        _pipeline = GenAISepsisPipeline()
        logger.info("GenAI Pipeline initialized")
    return _pipeline


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "GenAI Sepsis Classification API",
        "version": "1.0.0",
        "description": "AI-powered sepsis early warning system",
        "endpoints": {
            "classify": "POST /classify",
            "batch": "POST /classify-batch",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        pipeline = get_pipeline()
        health = pipeline.health_check()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipeline": health
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/classify")
async def classify_patient(
    request: ClassifyRequest,
    x_api_key: str = Header(None)
):
    """
    Classify a single patient for sepsis risk.
    
    Returns a risk score (0-100), priority level, and clinical rationale.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        
        # Run prediction
        result = pipeline.predict(
            patient_vitals=request.vitals,
            clinician_notes=request.notes,
            patient_id=request.patient_id
        )
        
        # Extract key fields for response
        final_pred = result.get("final_prediction", {})
        dashboard = result.get("dashboard_summary", {})
        logic_gate = result.get("logic_gate", {})
        
        return {
            "request_id": result.get("request_id"),
            "patient_id": result.get("patient_id"),
            "status": result.get("status", "success"),
            "risk_score": final_pred.get("risk_score_0_100", 0),
            "priority": final_pred.get("priority", "Standard"),
            "sepsis_probability_6h": final_pred.get("sepsis_probability_6h", "Low"),
            "clinical_rationale": final_pred.get("clinical_rationale", ""),
            "alert_level": dashboard.get("alert_level", "STANDARD"),
            "alert_color": dashboard.get("alert_color", "green"),
            "action_required": dashboard.get("action_required", "Routine monitoring"),
            "guardrail_override": logic_gate.get("guardrail_override", False),
            "override_reasons": logic_gate.get("override_reasons", []),
            "total_processing_time_ms": result.get("total_processing_time_ms", 0)
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-batch")
async def classify_batch(
    request: BatchClassifyRequest,
    x_api_key: str = Header(None)
):
    """
    Classify multiple patients in batch.
    
    Returns a list of classification results.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        
        # Prepare batch input
        patients = [
            {
                "vitals": p.vitals,
                "notes": p.notes,
                "patient_id": p.patient_id
            }
            for p in request.patients
        ]
        
        # Run batch prediction
        results = pipeline.predict_batch(patients)
        
        # Format responses
        responses = []
        for result in results:
            final_pred = result.get("final_prediction", {})
            dashboard = result.get("dashboard_summary", {})
            logic_gate = result.get("logic_gate", {})
            
            responses.append({
                "request_id": result.get("request_id"),
                "patient_id": result.get("patient_id"),
                "status": result.get("status", "success"),
                "risk_score": final_pred.get("risk_score_0_100", 0),
                "priority": final_pred.get("priority", "Standard"),
                "sepsis_probability_6h": final_pred.get("sepsis_probability_6h", "Low"),
                "clinical_rationale": final_pred.get("clinical_rationale", ""),
                "alert_level": dashboard.get("alert_level", "STANDARD"),
                "alert_color": dashboard.get("alert_color", "green"),
                "guardrail_override": logic_gate.get("guardrail_override", False),
                "batch_index": result.get("batch_index")
            })
        
        return {
            "status": "success",
            "total_patients": len(responses),
            "results": responses
        }
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/guardrail/thresholds")
async def get_guardrail_thresholds(x_api_key: str = Header(None)):
    """
    Get current guardrail threshold configuration.
    
    Returns all configurable clinical thresholds used by the safety guardrail.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        thresholds = pipeline.guardrail.get_current_thresholds()
        return {
            "status": "success",
            "thresholds": thresholds
        }
    except Exception as e:
        logger.error(f"Error getting thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/guardrail/reload")
async def reload_guardrail_config(x_api_key: str = Header(None)):
    """
    Hot-reload guardrail configuration.
    
    Reloads the genai_clinical_guardrail.json file without restarting the service.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        pipeline.guardrail.reload_config()
        return {
            "status": "success",
            "message": "Guardrail configuration reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/guardrail/config")
async def get_full_guardrail_config(x_api_key: str = Header(None)):
    """
    Get full guardrail configuration including all clinical details.
    
    Returns the complete configuration including:
    - Critical thresholds with clinical rationale, SME notes, and context checks
    - Early detection patterns
    - History context checks
    - Override logic
    - Discordance rules
    - qSOFA criteria
    - Audit settings
    
    This endpoint is designed for SME review and configuration management.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        config = pipeline.guardrail.get_full_config()
        return {
            "status": "success",
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting full config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GuardrailConfigUpdate(BaseModel):
    """Request model for updating guardrail configuration."""
    section: str = Field(..., description="Config section to update (e.g., 'critical_thresholds', 'override_logic', 'discordance_rules')")
    updates: Dict[str, Any] = Field(..., description="Updates to apply to the section")
    save_to_file: bool = Field(True, description="Whether to persist changes to genai_clinical_guardrail.json")


@app.put("/guardrail/config")
async def update_guardrail_config(
    request: GuardrailConfigUpdate,
    x_api_key: str = Header(None)
):
    """
    Update guardrail configuration.
    
    Allows SMEs to update specific sections of the guardrail configuration:
    - critical_thresholds: Update threshold values and clinical details
    - override_logic: Modify override trigger conditions and forced values
    - discordance_rules: Update concerning phrases and escalation settings
    - audit_settings: Configure logging behavior
    
    Changes can be applied in-memory only or persisted to the config file.
    
    Example request body:
    ```json
    {
        "section": "critical_thresholds.hemodynamic",
        "updates": {
            "sbp_critical": {
                "value": 85,
                "clinical_rationale": "Updated based on new guidelines"
            }
        },
        "save_to_file": true
    }
    ```
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        result = pipeline.guardrail.update_config(
            section=request.section,
            updates=request.updates,
            save_to_file=request.save_to_file
        )
        
        return {
            "status": "success",
            "message": f"Configuration section '{request.section}' updated successfully",
            "saved_to_file": request.save_to_file,
            "updated_values": result.get("updated_values", {}),
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/guardrail/config/export")
async def export_guardrail_config(x_api_key: str = Header(None)):
    """
    Export the full guardrail configuration as downloadable JSON.
    
    Use this endpoint to download the current configuration (including any
    in-memory changes) for manual saving to the source file.
    
    This is useful when the container filesystem is read-only and 
    save_to_file cannot be used.
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        config_json = pipeline.guardrail.export_config()
        
        return JSONResponse(
            content=json.loads(config_json),
            headers={
                "Content-Disposition": "attachment; filename=genai_clinical_guardrail.json"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/guardrail/config/{section}")
async def get_guardrail_config_section(
    section: str,
    x_api_key: str = Header(None)
):
    """
    Get a specific section of the guardrail configuration.
    
    Available sections:
    - critical_thresholds
    - early_detection_patterns
    - history_context_checks
    - differential_diagnosis
    - trending_requirements
    - override_logic
    - qsofa_criteria
    - discordance_rules
    - audit_settings
    - expert_notes
    
    Use dot notation for nested sections: critical_thresholds.hemodynamic
    """
    verify_api_key(x_api_key)
    
    try:
        pipeline = get_pipeline()
        config = pipeline.guardrail.get_full_config()
        
        # Handle nested sections like "critical_thresholds.hemodynamic"
        keys = section.split(".")
        result = config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Section '{section}' not found in configuration"
                )
        
        return {
            "status": "success",
            "section": section,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config section: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    logger.info("Starting GenAI Sepsis Classification API...")
    try:
        # Pre-initialize pipeline
        get_pipeline()
        logger.info("API startup complete")
    except Exception as e:
        logger.warning(f"Pipeline pre-initialization failed: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
