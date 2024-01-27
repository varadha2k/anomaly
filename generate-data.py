import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Function to generate random data
def generate_data(num_records):
    ids = np.arange(1, num_records + 1)
    company_names = [f"Company_{i}" for i in ids]
    start_dates = pd.date_range(start='2022-01-01', periods=num_records, freq='D').strftime('%Y-%m-%d')
    revenues = np.random.randint(100000, 1000000, size=num_records)
    lines_of_business = ['IT', 'Finance', 'Healthcare', 'Manufacturing', 'Retail']
    line_of_business = np.random.choice(lines_of_business, size=num_records)
    locations = ['City_A', 'City_B', 'City_C', 'City_D']
    location = np.random.choice(locations, size=num_records)
    profits = np.random.randint(-50000, 50000, size=num_records)

    # Introduce anomalies in 10 records
    anomaly_indices = np.random.choice(num_records, size=10, replace=False)
    revenues[anomaly_indices] = np.random.randint(1000000, 2000000, size=10)
    profits[anomaly_indices] = np.random.randint(-100000, -50000, size=10)

    data = {
        'ID': ids,
        'CompanyName': company_names,
        'StartDate': start_dates,
        'Revenue': revenues,
        'LineOfBusiness': line_of_business,
        'Location': location,
        'Profit': profits
    }
    
    df = pd.DataFrame(data)
    anomalous_records = df.loc[anomaly_indices]

    # Print anomalous records
    print("Anomalous Records:")
    print(anomalous_records)

    return df, anomaly_indices

# Generate 100 records
num_records = 100
dataset, anomaly_indices = generate_data(num_records)

# Save the dataset to a CSV file
dataset.to_csv('generated_dataset.csv', index=False)

# Separate normal and anomalous rows
normal_rows = dataset.drop(index=anomaly_indices)
anomalous_rows = dataset.loc[anomaly_indices]

# Include 10 normal records in the 'anomalous_records.csv' file
additional_normal_rows = normal_rows.sample(10)
anomalous_records_combined = pd.concat([anomalous_rows, additional_normal_rows])

# Save normal and anomalous records to separate files
normal_rows.to_csv('normal_records.csv', index=False)
anomalous_records_combined.to_csv('anomalous_records.csv', index=False)

# Print paths to the saved files
print("Normal Records saved to: normal_records.csv")
print("Anomalous Records saved to: anomalous_records.csv")