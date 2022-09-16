import json
import requests
import numpy as np

endpoint = "https://2izrpyb9f2.execute-api.us-west-1.amazonaws.com/"

# Generate sample data for prediction
sample = np.random.randn(1, 7)
sample_json = json.dumps(sample.tolist())

response = requests.post(endpoint, headers={"content-type": "application/json"}, data=sample_json)