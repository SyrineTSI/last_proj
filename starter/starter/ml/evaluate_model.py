from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data


def evaluate_model_slices(model_path, encoder_path, lb_path, test_data_path, categorical_features):
    # Load the trained model
    trained_model = joblib.load(model_path)

    encoder = joblib.load(encoder_path)
    lb  = joblib.load(lb_path)

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Process test data
    X_test, y_test, _, _ = process_data(
        test_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )

    performance_results = {}

    # Evaluate performance on slices of categorical features
    for feature in categorical_features:
        unique_values = test_data[feature].unique()
        for value in unique_values:
            # Create slice of data for current categorical value
            slice_mask = (test_data[feature] == value)
            X_slice = X_test[slice_mask]
            y_slice = y_test[slice_mask]

            # Make predictions
            y_pred = trained_model.predict(X_slice)

            # Calculate metrics
            precision = precision_score(y_slice, y_pred)
            recall = recall_score(y_slice, y_pred)
            f1 = f1_score(y_slice, y_pred)

            # Store results
            if feature not in performance_results:
                performance_results[feature] = {}
            performance_results[feature][value] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            # Print or log results
            print(f"Performance for {feature}='{value}':")
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
            print()

    return performance_results

# Example usage
if __name__ == "__main__":
    model_path = '/Users/A200226491/Desktop/Learning/last_proj/starter/starter/trained_model.pkl'
    encoder_path = '/Users/A200226491/Desktop/Learning/last_proj/starter/starter/encoder.pkl'
    lb_path = '/Users/A200226491/Desktop/Learning/last_proj/starter/starter/label_binarizer.pkl'
    test_data_path = '/Users/A200226491/Desktop/Learning/last_proj/census_cleaned.csv'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    results = evaluate_model_slices(model_path, 
                                    encoder_path,
                                    lb_path,
                                    test_data_path, 
                                    categorical_features)
