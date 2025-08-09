from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data(save_path="data/california.csv"):
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    load_data()
