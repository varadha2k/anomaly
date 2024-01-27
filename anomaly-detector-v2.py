import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import matplotlib.pyplot as plt


def visualize_anomalies(data):
    """
    Visualizes anomalies in the input data using a scatter plot.

    Parameters:
    - data: DataFrame, input data with 'index' and 'Revenue' columns

    Returns:
    - None
    """
    # Predict anomalies using the trained model
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data['Revenue'], c=anomaly_predictions, cmap='viridis', marker='o', label='Normal (1), Anomalous (-1)')
    plt.xlabel('Record Index')
    plt.ylabel('Revenue')
    plt.title('Isolation Forest Anomaly Detection')
    plt.legend()
    plt.savefig('anomaly_visualization.png')
    plt.show()
# Load the normal training data
normal_records = pd.read_csv('normal_records.csv')

# Select numeric columns for training the model
numeric_columns = normal_records.select_dtypes(include=['float64', 'int64']).columns
train_data = normal_records[numeric_columns]

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(train_data)

# Save the model to a pickle file
with open('anomaly_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the anomalous validation data
anomalous_records = pd.read_csv('anomalous_records.csv')

# Select numeric columns for validation
validation_data = anomalous_records[numeric_columns]

# Predict anomalies using the trained model
anomaly_predictions = model.predict(validation_data)
anomalous_records['Anomaly'] = anomaly_predictions

# Separate normal and anomalous records in the validation data
normal_records_validation = anomalous_records[anomalous_records['Anomaly'] == 1]
anomalous_records_validation = anomalous_records[anomalous_records['Anomaly'] == -1]

# Print normal records, anomalous records, and indicate anomalous columns
print("Normal Records in Validation:")
print(normal_records_validation)

print("\nAnomalous Records in Validation:")
print(anomalous_records_validation)

# Print columns marked as anomalous
anomalous_columns = anomalous_records_validation.drop('Anomaly', axis=1).apply(lambda x: x[x == -1].index.tolist(), axis=1)
print("\nAnomalous Columns within Rows:")
print(anomalous_columns)

visualize_anomalies(anomalous_records)