from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Hospital Readmission Prediction API")

# Load the trained model
model_data = joblib.load("readmission_model_improved.pkl")

model = model_data["model"]
feature_columns = model_data["feature_columns"]
threshold = model_data["threshold"]


class PatientData(BaseModel):
    age_mid: float
    time_in_hospital: int
    num_lab_procedures: int
    num_medications: int
    number_inpatient: int
    number_emergency: int
    number_outpatient: int
    meds_per_day: float

    adm_1: int
    adm_2: int
    adm_3: int
    adm_4: int
    adm_5: int
    adm_6: int
    adm_7: int
    adm_8: int


@app.get("/")
def home():
    return {"message": "Hospital Readmission Prediction API is running!"}


@app.post("/predict")
def predict(data: PatientData):

    # Convert input into a dictionary
    patient_data = data.model_dump()

    # Arrange features in the exact order expected by the model
    input_df = pd.DataFrame(
        [patient_data],
        columns=feature_columns
    )

    # Get prediction probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Get probability of class 1
    class_1_index = list(model.classes_).index(1)
    probability_class_1 = probabilities[class_1_index]

    # Apply saved threshold
    prediction = int(probability_class_1 >= threshold)

    return {
        "predicted_class": prediction,
        "class_0_probability": round(float(probabilities[0]), 4),
        "class_1_probability": round(float(probability_class_1), 4),
        "threshold": round(float(threshold), 4)
    }