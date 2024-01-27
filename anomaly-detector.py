# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your dataset (replace 'your_dataset.csv' with the actual file path or URL)
df = pd.read_csv('data.csv')

# Drop non-numeric columns that are not needed for anomaly detection
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]

# Standardize the data
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df_numeric)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df_numeric_scaled, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination based on your dataset
model.fit(train_data)

# Predict anomalies on the test set
df['Anomaly'] = model.predict(df_numeric_scaled)

# Identify rows and columns with anomalies
anomalous_rows = df[df['Anomaly'] == -1]
anomalous_columns = anomalous_rows.drop('Anomaly', axis=1).apply(lambda x: x[x == -1].index.tolist(), axis=1)

# Print the anomalous rows and columns
print("Anomalous Rows:")
print(anomalous_rows)

print("\nAnomalous Columns within Rows:")
print(anomalous_columns)
