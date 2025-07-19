from src.train import load_config, train_model
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train import load_config, train_model



def test_config_loading():
    config = load_config("config/config.json")
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    acc = model.score(X, y)
    assert acc > 0.9
