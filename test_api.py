from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_valid_data():
    valid_data = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [">50K"]