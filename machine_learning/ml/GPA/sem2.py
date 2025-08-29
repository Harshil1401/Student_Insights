import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("data/ml/synthetic_full_gpa_data.csv")

# Ensure all required GPA columns exist
required_columns = ['Sem1 GPA', 'Sem2 GPA']
df = df[required_columns].dropna()  # drop rows with missing GPAs

# Features (Sem1 to Sem7), Target (Sem8)
X = df[['Sem1 GPA']]
y = df['Sem2 GPA']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Compute RMSE (manual square root)
rmse = mean_squared_error(y_test, y_pred) ** 0.5



print(f"✅ R² Score: {r2:.2f}")
print(f"✅ RMSE: {rmse:.2f}")

model_data = {
    "model": model,
    "metrics": {
        "r2_score": r2,
        "rmse": rmse
    }
}

# Save the model
joblib.dump(model_data, "data/ml/Sem2_gpa_forecaster.pkl")
print("📦 Model saved as Sem2_gpa_forecaster.pkl")


# import joblib

# # Load model
# model = joblib.load("sem4_gpa_predictor.pkl")

# # Input: Sem1–4, Sem6–8 (exclude Sem5)
# input_gpa = [[7.1, 7.3, 7.5, 7.6, 7.9, 8.0, 8.1]]
# predicted = model.predict(input_gpa)

# print("🎯 Predicted Sem5 GPA:", predicted[0])


