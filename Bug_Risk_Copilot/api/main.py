"""
Phase 5.1-5.3: FastAPI Service
REST API for bug-risk predictions and explanations.

Endpoints:
  POST /predict - Get risk score from ML model
  POST /explain - Run agent explanation
  POST /analyze - End-to-end analysis (predict + explain)
  GET /health - Service health check
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import pickle
import numpy as np

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from agent.agent import BugRiskAgent

# Initialize FastAPI app
app = FastAPI(
    title="Bug-Risk Copilot",
    description="ML-powered bug risk prediction with agentic explanation",
    version="1.0.0"
)

# Enable CORS for GitHub Actions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models (Pydantic)
# ============================================================================

class DiffInput(BaseModel):
    """Input: Git diff or commit message."""
    diff: str = Field(..., description="Git diff content or commit message")
    commit_hash: Optional[str] = Field(None, description="Commit hash (for tracking)")


class PredictionResponse(BaseModel):
    """Output: Risk prediction from ML model."""
    commit_hash: Optional[str]
    risk_score: float = Field(..., description="Risk score 0-1")
    top_features: List[Dict] = Field(..., description="Top contributing factors (SHAP)")
    model_version: str


class ExplanationResponse(BaseModel):
    """Output: Agent explanation and routing."""
    commit_hash: Optional[str]
    explanation: str
    action: str = Field(..., description="'comment' or 'escalate'")
    confidence: float


class AnalysisResponse(BaseModel):
    """Output: Combined prediction + explanation."""
    commit_hash: Optional[str]
    risk_score: float
    explanation: str
    action: str
    confidence: float
    top_features: List[Dict]


# ============================================================================
# Model Loading and Initialization
# ============================================================================

MODEL_DIR = Path("models/artifacts")
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
FEATURE_COLS = None  # Will be loaded from trained model
model_cache = {}


def load_model():
    """Lazy load the trained model."""
    if "model" not in model_cache:
        try:
            if not MODEL_PATH.exists():
                logger.error(f"Model not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            
            with open(MODEL_PATH, 'rb') as f:
                model_cache["model"] = pickle.load(f)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return model_cache["model"]


def get_agent():
    """Get or create the LangGraph agent."""
    if "agent" not in model_cache:
        try:
            # Initialize agent (without Gemini API key for MVP)
            agent = BugRiskAgent(threshold=0.6)
            agent.build_graph()
            model_cache["agent"] = agent
            logger.info("Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    return model_cache["agent"]


# ============================================================================
# Helper Functions
# ============================================================================

def extract_features_from_diff(diff: str) -> Dict[str, float]:
    """
    Extract ML features from diff text.
    
    In production, this would parse the diff properly.
    For MVP, return mock features.
    """
    # Mock feature extraction
    lines = diff.split('\n')
    additions = sum(1 for line in lines if line.startswith('+'))
    deletions = sum(1 for line in lines if line.startswith('-'))
    files = len(set(line.split(' b/')[1] for line in lines if line.startswith('diff --git')))
    
    # Return features matching the trained model
    features = {
        'lines_added': min(additions, 500),
        'lines_deleted': min(deletions, 500),
        'total_churn': min(additions + deletions, 1000),
        'churn_ratio': additions / max(deletions, 1),
        'num_files_changed': min(files, 20),
        'num_directories_touched': max(1, files // 2),
        'num_subsystems_touched': max(1, files // 5),
        'max_complexity_delta': 0,
        'avg_complexity_delta': 0,
        'file_age_days': 365,
        'file_prior_bug_count': 0,
        'author_prior_bug_rate': 0.1,
        'author_commit_count': 50,
        'message_length': len(diff),
        'has_issue_reference': 1,
        'hour_of_day': 12,
        'day_of_week_num': 3,
    }
    
    return features


def predict_risk_score(features: Dict[str, float]) -> tuple:
    """
    Get risk prediction from model.
    
    Returns:
        (risk_score, top_features)
    """
    try:
        model = load_model()
        
        # Prepare feature array (match training order)
        feature_names = [
            'lines_added', 'lines_deleted', 'total_churn', 'churn_ratio',
            'num_files_changed', 'num_directories_touched', 'num_subsystems_touched',
            'max_complexity_delta', 'avg_complexity_delta', 'file_age_days',
            'file_prior_bug_count', 'author_prior_bug_rate', 'author_commit_count',
            'message_length', 'has_issue_reference', 'hour_of_day', 'day_of_week_num'
        ]
        
        X = np.array([[features.get(name, 0) for name in feature_names]], dtype=np.float32)
        
        # Get prediction
        risk_score = model['model'].predict_proba(X)[0, 1]
        
        # Mock SHAP features (in production, compute real SHAP values)
        top_features = [
            {'feature': 'total_churn', 'value': features['total_churn'], 'shap_value': 0.15},
            {'feature': 'file_prior_bug_count', 'value': features['file_prior_bug_count'], 'shap_value': 0.12},
            {'feature': 'num_files_changed', 'value': features['num_files_changed'], 'shap_value': 0.08},
        ]
        
        return float(risk_score), top_features
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bug-risk-copilot"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: DiffInput) -> PredictionResponse:
    """
    Predict bug risk from a diff.
    
    Runs the trained ML model to get risk score and feature importance.
    """
    try:
        logger.info(f"Predicting for {request.commit_hash or 'unknown'}")
        
        # Extract features
        features = extract_features_from_diff(request.diff)
        
        # Get prediction
        risk_score, top_features = predict_risk_score(features)
        
        response = PredictionResponse(
            commit_hash=request.commit_hash,
            risk_score=risk_score,
            top_features=top_features,
            model_version="xgboost-1.0"
        )
        
        logger.info(f"  Risk score: {risk_score:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain(prediction: PredictionResponse) -> ExplanationResponse:
    """
    Generate explanation for a prediction.
    
    Runs the LangGraph agent to explain the ML model's decision.
    """
    try:
        logger.info(f"Explaining prediction for {prediction.commit_hash or 'unknown'}")
        
        agent = get_agent()
        
        # Run agent
        state = {
            'commit_hash': prediction.commit_hash or 'unknown',
            'diff_text': 'commit details',
            'risk_score': prediction.risk_score,
            'top_features': prediction.top_features,
        }
        
        result = agent.run(state)
        
        response = ExplanationResponse(
            commit_hash=prediction.commit_hash,
            explanation=result.get('explanation', 'No explanation available'),
            action=result.get('action', 'comment'),
            confidence=result.get('confidence', 0.5)
        )
        
        logger.info(f"  Action: {response.action}")
        return response
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: DiffInput) -> AnalysisResponse:
    """
    End-to-end analysis: predict + explain.
    
    This is the main endpoint for integrating with GitHub Actions.
    """
    try:
        logger.info(f"Analyzing {request.commit_hash or 'unknown'}")
        
        # Step 1: Predict
        features = extract_features_from_diff(request.diff)
        risk_score, top_features = predict_risk_score(features)
        
        # Step 2: Explain
        agent = get_agent()
        state = {
            'commit_hash': request.commit_hash or 'unknown',
            'diff_text': request.diff,
            'risk_score': risk_score,
            'top_features': top_features,
        }
        
        result = agent.run(state)
        
        response = AnalysisResponse(
            commit_hash=request.commit_hash,
            risk_score=risk_score,
            explanation=result.get('explanation', 'No explanation available'),
            action=result.get('action', 'comment'),
            confidence=result.get('confidence', 0.5),
            top_features=top_features
        )
        
        logger.info(f"  Risk: {risk_score:.3f}, Action: {response.action}")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Server Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup."""
    try:
        logger.info("Starting up server...")
        load_model()
        get_agent()
        logger.info("Server ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
