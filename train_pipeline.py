import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import yaml
from preprocessing import RecategorizeSmoking, GenderCleaner

DATA_DIR = Path('.')
TRAIN_PATH = DATA_DIR / 'train.csv'
VALID_PATH = DATA_DIR / 'valid.csv'
TEST_PATH = DATA_DIR / 'test.csv'
MODEL_PATH = DATA_DIR / 'final_xgboost_model.pkl'

TARGET = 'diabetes'
FEATURES = [
    'gender',
    'age',
    'hypertension',
    'heart_disease',
    'smoking_history',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
]

class RecategorizeSmoking(BaseEstimator, TransformerMixin):
    """
    Deprecated inline definition. Use preprocessing.RecategorizeSmoking instead.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, valid_df, test_df


def build_pipeline():
    cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    categories = [
        ['Female', 'Male'],
        [0, 1],
        [0, 1],
        ['current_smoker', 'non_smoker', 'past_smoker', 'unknown'],
    ]

    preprocess = Pipeline([
        ('gender_clean', GenderCleaner(baseline='Female')),
        ('recategorize', RecategorizeSmoking()),
        ('encoder', ColumnTransformer([
            ('cat', OneHotEncoder(categories=categories, drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
        ], remainder='passthrough', verbose_feature_names_out=False)),
    ])

    base = Pipeline([
        ('preprocess', preprocess),
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss')),
    ])

    # Probability calibration using isotonic with internal CV
    calibrated = CalibratedClassifierCV(base, cv=5, method='isotonic')
    return calibrated


def main():
    train_df, valid_df, test_df = load_data()

    # Use training set for fitting; validation/test reserved for evaluation
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Cross-fold threshold optimization on training data (stratified 5-fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    threshold_grid = np.linspace(0.1, 0.9, 81)
    grid_scores = {thr: [] for thr in threshold_grid}
    fold_best_thresholds = []
    fold_best_f1s = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        pipe_cv = build_pipeline()
        pipe_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        val_proba_cv = pipe_cv.predict_proba(X_train.iloc[val_idx])[:, 1]
        best_thr_fold, best_f1_fold = None, -1
        for thr in threshold_grid:
            preds = (val_proba_cv >= thr).astype(int)
            score = f1_score(y_train.iloc[val_idx], preds)
            grid_scores[thr].append(score)
            if score > best_f1_fold:
                best_f1_fold = score
                best_thr_fold = thr
        fold_best_thresholds.append(best_thr_fold)
        fold_best_f1s.append(best_f1_fold)

    avg_scores = {thr: float(np.mean(scores)) for thr, scores in grid_scores.items()}
    best_thr = max(avg_scores, key=avg_scores.get)
    best_f1_mean = avg_scores[best_thr]
    best_f1_std = float(np.std(grid_scores[best_thr]))
    print(f"Cross-fold threshold selected: {best_thr:.2f} (mean F1={best_f1_mean:.3f}, std={best_f1_std:.3f})")

    # Evaluation with selected threshold (use full-model fitted above)
    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]
    valid_pred = (valid_proba >= best_thr).astype(int)
    test_pred = (test_proba >= best_thr).astype(int)
    print(f"Validation F1: {f1_score(y_valid, valid_pred):.3f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_valid, valid_proba):.3f}")
    print(f"Test F1: {f1_score(y_test, test_pred):.3f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")

    # Save model and metadata (YAML)
    joblib.dump(pipe, MODEL_PATH)
    meta = {
        'model_version': '1.1.0',
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'target': TARGET,
        'features': FEATURES,
        'calibration': {'method': 'isotonic', 'cv': 5},
        'threshold': float(best_thr),
        'threshold_cv': {
            'n_splits': 5,
            'grid_min': 0.1,
            'grid_max': 0.9,
            'grid_step_count': 81,
            'fold_best_thresholds': [float(t) for t in fold_best_thresholds],
            'fold_best_f1s': [float(s) for s in fold_best_f1s],
            'mean_f1_at_selected_thr': float(best_f1_mean),
            'std_f1_at_selected_thr': float(best_f1_std),
        },
        'encoder': {
            'drop': 'first',
            'categories': {
                'gender': ['Female', 'Male'],
                'hypertension': [0, 1],
                'heart_disease': [0, 1],
                'smoking_history': ['current_smoker', 'non_smoker', 'past_smoker', 'unknown'],
            },
            'handle_unknown': 'ignore',
        },
    }
    with open(DATA_DIR / 'model_meta.yaml', 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"Saved unified pipeline to {MODEL_PATH} and metadata to model_meta.yaml")


if __name__ == '__main__':
    main()