import pandas as pd

# Load dataset
df = pd.read_csv("Dataset/customer_churn.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove customerID column (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove null values
df.dropna(inplace=True)

# Final dataset shape
print("\nFinal dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv("Dataset/cleaned_customer_churn.csv", index=False)

print("\nData cleaning completed successfully.")