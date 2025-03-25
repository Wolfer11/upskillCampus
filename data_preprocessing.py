import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None, None
    
    try:
        # Load dataset
        df = pd.read_csv(file_path, parse_dates=['DateTime'])  
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
    
    # Debug: Check column names
    print(f"Columns in dataset ({file_path}):", df.columns)
    
    if 'DateTime' not in df.columns:
        print(f"Error: 'DateTime' column missing in {file_path}")
        return None, None
    
    # Convert 'DateTime' to datetime and sort
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.sort_values('DateTime', inplace=True)
    
    # Feature Engineering
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_holiday'] = df['DateTime'].dt.strftime('%Y-%m-%d').isin(['2025-01-01', '2025-12-25']).astype(int)
    
    # Scaling
    scaler = MinMaxScaler()
    target_column = 'Vehicles'  # Adjust based on your dataset
    
    if target_column in df.columns:
        df[[target_column]] = scaler.fit_transform(df[[target_column]])
    else:
        print(f"Warning: Column '{target_column}' not found in dataset {file_path}. Skipping scaling.")
    
    return df, scaler

# File paths
file_paths = [
    r"E:\Smart city forecasting\Project9_smart-city-traffic-patterns\Project9_smart-city-traffic-patterns\smart-city-traffic-patterns\datasets_8494_11879_test_BdBKkAj.csv",
    r"E:\Smart city forecasting\Project9_smart-city-traffic-patterns\Project9_smart-city-traffic-patterns\smart-city-traffic-patterns\traffic_data.csv"
]

# Process each file
for file in file_paths:
    df, scaler = preprocess_data(file)
    if df is not None:
        print(df.head())  # Print first few rows for verification