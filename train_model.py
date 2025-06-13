import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("loan.csv")  # Make sure loan.csv is in the same folder

# Drop NA
df = df.dropna()


# Clean 'Dependents' column: convert '3+' to 3
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
# Convert categorical to numeric

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

# Assuming Loan_Status column is manually added for now
# For demo, simulate it
import numpy as np
df['Loan_Status'] = np.random.randint(0, 2, size=len(df))

# Define X and y
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
