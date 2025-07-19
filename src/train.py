import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def load_config(path):
    with open(path) as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

def main():
    config = load_config("config/config.json")
    data = load_digits()
    X, y = data.data, data.target
    model = train_model(X, y, config)
    joblib.dump(model, "model_train.pkl")

if __name__ == "__main__":
    main()
