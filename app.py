import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

model   = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols   = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Serve the frontend
@app.get("/")
def index():
    return FileResponse("templates/index.html")


class StrokeRequest(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(req: StrokeRequest):
    try:
        raw = pd.DataFrame([{
            "age": req.age,
            "hypertension": req.hypertension,
            "heart_disease": req.heart_disease,
            "avg_glucose_level": req.avg_glucose_level,
            "bmi": req.bmi,
            "gender": req.gender,
            "ever_married": req.ever_married,
            "work_type": req.work_type,
            "Residence_type": req.Residence_type,
            "smoking_status": req.smoking_status,
        }])

        ohe_array = encoder.transform(raw[categorical_cols])
        ohe_df    = pd.DataFrame(ohe_array, columns=encoder.get_feature_names_out(categorical_cols))

        final = pd.concat([raw[numerical_cols].reset_index(drop=True), ohe_df], axis=1)

        prediction  = int(model.predict(final)[0])
        probability = model.predict_proba(final)[0].tolist()

        return {
            "stroke_prediction": prediction,
            "probability": {
                "no_stroke": round(probability[0], 4),
                "stroke":    round(probability[1], 4),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
