import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RecategorizeSmoking(BaseEstimator, TransformerMixin):
    """Custom transformer to recategorize smoking_history into
    ['non_smoker', 'current_smoker', 'past_smoker', 'unknown'] without changing row count.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def map_smoking(v):
            if pd.isna(v):
                return 'unknown'
            s = str(v).strip().lower().replace(' ', '_')
            if s in {'never', 'non_smoker'}:
                return 'non_smoker'
            elif s in {'current', 'current_smoker'}:
                return 'current_smoker'
            elif s in {'former', 'past_smoker', 'ever', 'not_currently', 'not_current'}:
                return 'past_smoker'
            elif s in {'no_info', 'unknown'}:
                return 'unknown'
            elif s in {'non_smoker', 'current_smoker', 'past_smoker'}:
                return s
            else:
                return 'unknown'
        X['smoking_history'] = X['smoking_history'].apply(map_smoking)
        return X

class GenderCleaner(BaseEstimator, TransformerMixin):
    """Normalize gender values; map 'Other' to 'Female' (baseline) for consistency.
    If value is missing or unrecognized, default to 'Female'.
    """
    def __init__(self, baseline='Female'):
        self.baseline = baseline

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def map_gender(v):
            if pd.isna(v):
                return self.baseline
            s = str(v).strip().capitalize()
            if s in {'Female', 'Male'}:
                return s
            # unify 'Other' or anything else to baseline
            return self.baseline
        X['gender'] = X['gender'].apply(map_gender)
        return X