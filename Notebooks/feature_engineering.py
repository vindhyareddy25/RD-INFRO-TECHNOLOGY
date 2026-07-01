import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("Dataset/cleaned_customer_churn.csv")

# Check object columns
print("Categorical columns:")
print(df.select_dtypes(include=['object']).columns)

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Verify all columns are numeric
print("\nData types after encoding:")
print(df.dtypes)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save dataset
df.to_csv("Dataset/feature_engineered_churn.csv", index=False)

print("\nFeature Engineering completed successfully.")
print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)