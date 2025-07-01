from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
from src.api.pydantic_models import PredictRequest, PredictResponse

app = FastAPI(title="Credit Risk Prediction API")

# Load model from MLflow registry
try:
    model = mlflow.pyfunc.load_model(model_uri="models:/credit-risk-model/Production")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API is up and running!"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Convert Pydantic input into a pandas DataFrame
        input_df = pd.DataFrame([request.dict()])
        prediction = model.predict(input_df)
        probability = float(prediction[0])  # model returns array-like
        return PredictResponse(fraud_probability=probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
