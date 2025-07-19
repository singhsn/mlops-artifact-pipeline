import joblib
from sklearn.datasets import load_digits

def inference():
    model = joblib.load("model_train.pkl")
    digits = load_digits()
    X = digits.data
    preds = model.predict(X)
    print("Sample Predictions:", preds[:10])

if __name__ == "__main__":
    inference()
