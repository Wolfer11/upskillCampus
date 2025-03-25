import requests
import numpy as np

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample input sequence (Adjust as per your model's expected input shape)
data = {"sequence": np.random.rand(24).tolist()}  # Simulating a 24-hour traffic pattern

# Send a POST request
response = requests.post(url, json=data)

# Print the prediction result
print(response.json())  
