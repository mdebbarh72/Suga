from fastapi import FastAPI
from app.api.schemas import PatientInput
from app.api.predictor import predictRisk, loadModel

import mlflow
import src.utils.mlflowconf

from contextlib import asynccontextmanager
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    risk: str
    risk_level: int          # 1 = high, 0 = low
    badge_color: str         # "red" or "green"
    advice: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = loadModel()
    yield

app  = FastAPI(
    lifespan=lifespan, 
    title= 'diabetes risk predictor',
    description='api that predicts if a patient has diabetes or not',
    version = '1.0.0'
)


@app.get("/")
def root():
    return {"message": "Diabetes Risk API is running"}

@app.get('/health')
def health():
    return {
        'status': "ok"
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(patient: PatientInput):

    risk = predictRisk(patient.model_dump())
    if risk == "High":
        return PredictionResponse(
            risk="High", risk_level=1, badge_color="red",
            advice="Consult a doctor. Monitor glucose levels closely."
        )
    return PredictionResponse(
        risk="Normal", risk_level=0, badge_color="green",
        advice="Keep up healthy habits. Regular check-ups recommended."
    )
