# scripts/train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv('data/service_requests.csv')

# Features used
text_col = 'issue_text'
numeric_cols = ['car_age_years','km_driven','last_service_months']

# Preprocessing: simple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Create pipelines
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))

def get_text(x):
    return x[text_col].astype(str)

# numeric transformer
def get_numeric(x):
    return x[numeric_cols].fillna(0).values

from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key): self.key = key
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.key].astype(str).values

class NumSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys): self.keys = keys
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.keys].fillna(0).values

from sklearn.pipeline import FeatureUnion

text_pipe = Pipeline([
    ('selector', TextSelector(text_col)),
    ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1,2)))
])

num_pipe = Pipeline([
    ('selector', NumSelector(numeric_cols)),
    ('scaler', StandardScaler())
])

full_features = FeatureUnion([('text', text_pipe), ('num', num_pipe)])

X = df[[text_col] + numeric_cols]
# Service category (classifier)
y_cat = df['service_category']
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

clf_pipeline = Pipeline([
    ('features', full_features),
    ('clf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
])
clf_pipeline.fit(X_train, y_train_cat)
print("Service category classifier score:", clf_pipeline.score(X_test, y_test_cat))

# Priority classifier
y_pri = df['priority_level']
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X, y_pri, test_size=0.2, random_state=42)
pri_pipeline = Pipeline([
    ('features', full_features),
    ('clf', RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))
])
pri_pipeline.fit(Xp_train, yp_train)
print("Priority classifier score:", pri_pipeline.score(Xp_test, yp_test))

# Duration regressor
y_dur = df['estimated_hours']
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X, y_dur, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
dur_pipeline = Pipeline([
    ('features', full_features),
    ('reg', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])
dur_pipeline.fit(Xd_train, yd_train)
print("Duration R^2:", dur_pipeline.score(Xd_test, yd_test))

# save models
import os
os.makedirs('models', exist_ok=True)
joblib.dump(clf_pipeline, 'models/service_category_model.pkl')
joblib.dump(pri_pipeline, 'models/priority_model.pkl')
joblib.dump(dur_pipeline, 'models/duration_model.pkl')
print("Saved models in models/")
