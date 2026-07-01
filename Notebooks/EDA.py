import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Dataset/cleaned_customer_churn.csv")

# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("Task4_EDA/churn_distribution.png")
plt.show()

# Tenure distribution
plt.figure(figsize=(6,4))
sns.histplot(df["tenure"], bins=30)
plt.title("Tenure Distribution")
plt.savefig("Task4_EDA/tenure_distribution.png")
plt.show()

# Monthly charges vs churn
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("Task4_EDA/monthlycharges_vs_churn.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("Task4_EDA/correlation_heatmap.png")
plt.show()

print("EDA completed successfully.")