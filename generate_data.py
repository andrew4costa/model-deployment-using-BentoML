import pandas as pd
from sklearn.datasets import make_classification

def generate_data(n_samples, n_features, path):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5)
    
    # Save it as a CSV
    feature_names = [f"feature {i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    df.to_csv(path, index=False)

if __name__ == "__main__":
    n_samples, n_features = 10000, 7
    generate_data(n_samples, n_features, path="/Users/andrewcosta/Desktop/API/data.csv")