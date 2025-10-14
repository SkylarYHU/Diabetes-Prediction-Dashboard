# Diabetes Prediction Dashboard

A Streamlit dashboard for binary classification of diabetes risk using an XGBoost model. It provides an interactive Predict page, fairness analysis visuals, and simple model evaluation utilities.

## Features
- Interactive sidebar inputs (gender, age, hypertension, heart disease, smoking history, BMI, HbA1c, glucose).
- Default prediction shown on first load with a top info banner; users adjust inputs and click Predict to run their own inference.
- Two Predict buttons in the sidebar (one under the Input header, one at the bottom) for convenience.
- Adjustable classification threshold (default comes from model_meta.yaml).
- Fairness page with Altair bar charts visualizing performance by Gender and Smoking History (horizontal x-axis labels).
- Teaching/demo disclaimer to emphasize non-clinical use.

## Repository Structure
- `app.py` — Streamlit application.
- `preprocessing.py` — Preprocessing utilities (e.g., recategorization).
- `train_pipeline.py` — Training script for XGBoost model, data split, and metadata export.
- `final_xgboost_model.pkl` — Trained model artifact.
- `model_meta.yaml` — Model metadata (e.g., threshold).
- `cleaned_diabetes_dataset.csv`, `diabetes_prediction_dataset.csv` — Source/cleaned datasets.
- `train.csv`, `valid.csv`, `test.csv` — Data splits for training/validation/test.
- `requirements.txt` — Python dependencies.
- `Diabetes_Prediction.ipynb` — Notebook (exploration/experiments).
- `Diabetes_Prediction.html` — Project report/notes.

## Requirements
- Python 3.10+ (tested on 3.12).
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart (Local)
1. Ensure datasets and model files are available (provided in this repo).
2. Run the app:

```bash
streamlit run app.py
```

3. In the browser:
   - On first load, the page shows a default prediction (info banner at the top explains this).
   - Adjust parameters in the left sidebar and click the Predict button to run your own inference.
   - Use the threshold slider to explore sensitivity-specificity trade-offs.

## Pages Overview
- Predict:
  - Displays diabetes probability, current threshold, and a high/low risk message.
  - "View Input Features" expander shows the exact feature row passed to the model.
- Fairness:
  - Altair bar charts (with horizontal x-axis labels) for groups such as Gender and Smoking History.
  - Intended for teaching discussions of TPR/FPR and subgroup parity (extend as needed).

## Training
- The `train_pipeline.py` script trains the XGBoost model and writes the model artifact and `model_meta.yaml` (including an optimized threshold based on validation metrics).
- You can retrain using your own data (ensure consistent schema with the app’s preprocessing).

## Configuration
- Default threshold is read from `model_meta.yaml`. If absent, the app falls back to an internal default.
- All inputs are provided via the sidebar; the app computes features and calls `predict_proba` when available.

## Data & Privacy
- The included CSV files are for educational/demo purposes. Do not commit any real patient-identifiable data.
- If using real-world data, ensure proper de-identification and compliance with relevant regulations before training or sharing.

## Tips
- To keep the repository lightweight, consider using Git LFS or excluding large datasets/models.
- Add environment-specific secrets to `.streamlit/secrets.toml` (not committed) if you integrate external services.

## Contribution
Issues and pull requests are welcome. Please describe proposed changes clearly and include minimal reproducible examples where relevant.

## Disclaimer
This application is for teaching and demonstration purposes only and does not replace professional medical advice.