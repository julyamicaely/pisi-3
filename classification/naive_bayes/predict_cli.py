import argparse
import pandas as pd
import joblib
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def predict_from_cli(args):
    """Makes a prediction based on CLI arguments."""
    base_path = Path(__file__).parent.parent
    models_path = base_path / "models"
    scalers_path = base_path / "scalers"

    try:
        # Load model and preprocessor
        model = joblib.load(models_path / "naive_bayes_model.joblib")
        preprocessor = joblib.load(scalers_path / "naive_bayes_preprocessor.joblib")
    except FileNotFoundError as e:
        print(f"Error: Model or preprocessor file not found. {e}")
        print("Please ensure you have run the training script `train_naive_bayes.py` first.")
        return

    # Calculate BMI
    if args.height > 0:
        bmi = args.weight / ((args.height / 100) ** 2)
    else:
        bmi = 0

    # Create a DataFrame from the inputs
    feature_names = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    input_data = [
        args.age_years, args.height, args.weight, args.ap_hi, args.ap_lo, bmi,
        args.gender, args.cholesterol, args.gluc, args.smoke, args.alco, args.active
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Preprocess the data and make prediction
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    # Display result
    result_text = "Cardiovascular Disease DETECTED" if prediction[0] == 1 else "NO Cardiovascular Disease Detected"
    confidence = prediction_proba[0][prediction[0]] * 100

    print("\n--- Naive Bayes Prediction Result ---")
    print(f"Prediction: {result_text}")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cardiovascular disease using a trained Naive Bayes model.")
    
    # Add arguments for each feature
    parser.add_argument("--age-years", type=int, required=True, help="Age in years.")
    parser.add_argument("--height", type=int, required=True, help="Height in cm.")
    parser.add_argument("--weight", type=float, required=True, help="Weight in kg.")
    parser.add_argument("--ap-hi", type=int, required=True, help="Systolic blood pressure.")
    parser.add_argument("--ap-lo", type=int, required=True, help="Diastolic blood pressure.")
    parser.add_argument("--gender", type=int, choices=[1, 2], required=True, help="Gender (1: Female, 2: Male).")
    parser.add_argument("--cholesterol", type=int, choices=[1, 2, 3], required=True, help="Cholesterol (1: Normal, 2: Above Normal, 3: Well Above Normal).")
    parser.add_argument("--gluc", type=int, choices=[1, 2, 3], required=True, help="Glucose (1: Normal, 2: Above Normal, 3: Well Above Normal).")
    parser.add_argument("--smoke", type=int, choices=[0, 1], required=True, help="Smoker (0: No, 1: Yes).")
    parser.add_argument("--alco", type=int, choices=[0, 1], required=True, help="Alcohol intake (0: No, 1: Yes).")
    parser.add_argument("--active", type=int, choices=[0, 1], required=True, help="Physical activity (0: No, 1: Yes).")

    args = parser.parse_args()
    predict_from_cli(args)
