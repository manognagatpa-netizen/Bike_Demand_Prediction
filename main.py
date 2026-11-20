import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
BASE_DIR = Path(__file__).parent

# Model saved by your notebook: joblib.dump(random_forest_model, "random_forest_bike_demand.pkl")
model = joblib.load("random_forest_bike_demand.pkl")
# Load clustering model, scaler and labels
cluster_data = joblib.load("clustering_bike_demand.joblib")

cluster_model = cluster_data["model"]
cluster_scaler = cluster_data["scaler"]
cluster_features = cluster_data["features"]
cluster_labels = cluster_data["label_map"]
FEATURES = list(model.feature_names_in_)  # exact column order used in training

app = FastAPI(title="Bike Rental Demand API")

# Allows any website / app to call the API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class BikeInput(BaseModel):
    yr: int = Field(..., ge=0, le=1, description="0 = 2011, 1 = 2012")
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6, description="0 = Sunday")
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4, description="1 clear, 2 mist, 3 light rain/snow, 4 heavy rain/snow")
    season: int = Field(..., ge=1, le=4, description="1 winter, 2 spring, 3 summer, 4 fall")
    temp: float = Field(..., ge=0, le=1, description="Normalized temperature (0-1)")
    hum: float = Field(..., ge=0, le=1, description="Normalized humidity (0-1)")
    windspeed: float = Field(..., ge=0, le=1, description="Normalized wind speed (0-1)")

    model_config = {
        "json_schema_extra": {
            "example": {"yr": 1, "mnth": 9, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1,
                        "weathersit": 1, "season": 3, "temp": 0.6, "hum": 0.5, "windspeed": 0.2}
        }
    }


def build_features(d: dict) -> pd.DataFrame:
    """Repeats the exact feature engineering used in the notebook."""
    row = {
        "yr": d["yr"], "mnth": d["mnth"], "hr": d["hr"], "holiday": d["holiday"],
        "weekday": d["weekday"], "workingday": d["workingday"],
        "temp": d["temp"], "hum": d["hum"], "windspeed": d["windspeed"],
        "hr_sin": np.sin(2 * np.pi * d["hr"] / 24),
        "hr_cos": np.cos(2 * np.pi * d["hr"] / 24),
        "mnth_sin": np.sin(2 * np.pi * d["mnth"] / 12),
        "mnth_cos": np.cos(2 * np.pi * d["mnth"] / 12),
        "rush_hour": int(d["hr"] in [7, 8, 9, 16, 17, 18, 19]),
    }
    for k in (1, 2, 3, 4):
        row[f"wx_{k}"] = int(d["weathersit"] == k)
        row[f"ssn_{k}"] = int(d["season"] == k)
    return pd.DataFrame([row])[FEATURES]


@app.get("/")
def home():
    return {"message": "Bike Rental Demand API is running. Open /docs to try it."}


@app.post("/predict")
def predict(data: BikeInput):
    X = build_features(data.model_dump())
    pred = float(np.clip(model.predict(X)[0], 0, None))
    return {"predicted_rentals": round(pred)}
@app.post("/cluster")
def predict_cluster(data: BikeInput):

    # Get Random Forest demand prediction
    X = build_features(data.model_dump())

    predicted_demand = float(
        np.clip(model.predict(X)[0], 0, None)
    )

    # Prepare the four clustering features
    cluster_input = pd.DataFrame([{
        "avg_cnt": predicted_demand,
        "avg_temp": data.temp,
        "avg_hum": data.hum,
        "hr": data.hr
    }])

    # Ensure correct feature order
    cluster_input = cluster_input[cluster_features]

    # Apply the previously fitted scaler
    scaled_input = cluster_scaler.transform(cluster_input)

    # Predict cluster
    cluster_id = int(cluster_model.predict(scaled_input)[0])

    # Get demand level
    demand_level = cluster_labels[cluster_id]

    return {
        "predicted_rentals": round(predicted_demand),
        "cluster_id": cluster_id,
        "demand_level": demand_level
    }