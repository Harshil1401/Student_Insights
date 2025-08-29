import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/ml/synthetic_placement_dataset.csv")

# Target
y = df['CTC (LPA)']

# Features
features = [
    'Gender', 'Branch', 'Average GPA', 'Backlogs', 'Attendance (%)',
    'Skills', 'Certifications', 'Internship Domain', 'Job Role',
    'Company Tier', 'English Proficiency Score', 'Hackathons Participated',
    'Interview Performance (1-10)', 'Soft Skills Rating (1-10)', 'Offer Type'
]
X = df[features]

# Categorical & numerical split
categorical_features = [
    'Gender', 'Branch', 'Skills', 'Certifications', 'Internship Domain',
    'Job Role', 'Company Tier', 'Offer Type'
]
numerical_features = [
    'Average GPA', 'Backlogs', 'Attendance (%)', 'English Proficiency Score',
    'Hackathons Participated', 'Interview Performance (1-10)', 'Soft Skills Rating (1-10)'
]

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# Model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save model + accuracy
model_data = {
    "model": pipeline,
    "metrics": {
        "r2_score": r2,
        "mae": mae
    }
}

joblib.dump(model_data, "ctc_predictor.pkl")

print(f"Model saved with R²: {r2:.4f}, MAE: {mae:.4f}")
