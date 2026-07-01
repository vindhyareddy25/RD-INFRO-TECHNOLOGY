import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load feature engineered dataset
df = pd.read_csv("Dataset/feature_engineered_churn.csv")

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "Models/churn_model.pkl")

print("Model training completed successfully.")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)