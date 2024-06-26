# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
import os

from ml.data import process_data
from ml.model import train_model


#ml_dir = '/home/runner/work/last_proj/last_proj/starter/'
ml_dir = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(ml_dir, "trained_model.pkl")
ENCODER_PATH = os.path.join(ml_dir, "encoder.pkl")
LB_PATH = os.path.join(ml_dir, "label_binarizer.pkl")

# Add code to load in the data.
data = pd.read_csv('/Users/A200226491/Desktop/Learning/last_proj/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder, 
    lb=lb
)

# Train and save a model.
trained_model = train_model(X_train, y_train)

joblib.dump(encoder, ENCODER_PATH)
joblib.dump(lb, LB_PATH)

joblib.dump(trained_model, TRAINED_MODEL_PATH)
