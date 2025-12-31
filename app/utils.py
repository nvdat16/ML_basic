import pickle
import random
import numpy as np
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
