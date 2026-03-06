from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import model_helper
from typing import Any

app = FastAPI(title="CodeX Price Prediction API")

artifacts = None

@app.on_event("startup")
def load_artifacts():
    global artifacts
    try:
        artifacts = model_helper.load_model_artifacts()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18)
    gender: str
    zone: str
    occupation: str
    income_levels: str
    consume_frequency_weekly: str = Field(..., alias="consume_frequency(weekly)")
    current_brand: str
    preferable_consumption_size: str
    awareness_of_other_brands: str
    reasons_for_choosing_brands: str
    flavor_preference: str
    purchase_channel: str
    packaging_preference: str
    health_concerns: str
    typical_consumption_situations: str

    class Config:
        populate_by_name = True

@app.post("/predict")
def predict(request: PredictionRequest):
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    raw_input = request.model_dump(by_alias=True) if hasattr(request, "model_dump") else request.dict(by_alias=True)
    
    try:
        result = model_helper.predict_price_range(raw_input, artifacts)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
