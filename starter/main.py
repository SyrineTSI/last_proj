# Put the code for your API here.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import uvicorn
import os
import numpy as np

# Initialize FastAPI
app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and encoders
TRAINED_MODEL_PATH = base_dir / 'trained_model.pkl'
encoder_path = base_dir / 'encoder.pkl'
lb_path = base_dir / 'label_binarizer.pkl'
model = joblib.load(TRAINED_MODEL_PATH)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)


# Define Pydantic model for POST request body
class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Welcome message endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Inference API"}

# Model inference endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert the received data into a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Process the data similarly to how it was done during training
        X_categorical = encoder.transform(input_data[["workclass", 
                                                      "education", 
                                                      "marital_status", 
                                                      "occupation", 
                                                      "relationship", 
                                                      "race", 
                                                      "sex", 
                                                      "native_country"]])
        
        X_continuous = input_data.drop(columns=["workclass", 
                                                "education", 
                                                "marital_status",
                                                "occupation", 
                                                "relationship", 
                                                "race", 
                                                "sex", 
                                                "native_country"])

        X_processed = np.concatenate([X_continuous.values, X_categorical], 
                                     axis=1)

        # Perform inference using the trained model
        predictions = model.predict(X_processed)
        predicted_class = lb.inverse_transform(predictions)[0]

        return {"prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

