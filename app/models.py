from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_model(name):
    if name == "lr":
        return LogisticRegression(max_iter=1000)
    elif name == "svm":
        return SVC(probability=True)
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200)
    else:
        raise ValueError("Unknown model")
