import pandas as pd
import xgboost as xgb
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('intersection_features_target_variable_encoded.csv')
selected_features = [
    'ICD-O-3 Hist/behav, malignant', 'ER Status Recode Breast Cancer (1990+)',
    'Regional nodes positive (1988+)', 'Vital status recode (study cutoff used)',
    'Regional nodes examined (1988+)', 'Breast - Adjusted AJCC 6th T (1988-2015)',
    'Year of diagnosis', 'Reason no cancer-directed surgery',
    'Site recode - rare tumors', 'ICCC site recode extended 3rd edition/IARC 2017',
    'SEER cause-specific death classification', 'Derived HER2 Recode (2010+)',
    'COD to site recode ICD-O-3 2023 Revision Expanded (1999+)',
    'Breast Subtype (2010+)', 'AYA site recode 2020 Revision',
    'PR Status Recode Breast Cancer (1990+)'
]
X = df[selected_features]
y = df['Survival_months_graded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

with tqdm(total=1, desc="Training XGBoost") as pbar:
    model.fit(X_train, y_train)
    pbar.update(1)

explainer = shap.TreeExplainer(model)
with tqdm(total=1, desc="Calculating SHAP values") as pbar:
    shap_values = explainer.shap_values(X_test)
    pbar.update(1)
