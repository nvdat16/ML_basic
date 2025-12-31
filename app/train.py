import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocess import load_data, build_preprocessor

from utils import save_model
from models import get_model


def main(model):
    DATA_PATH = "dataset/Churn_Modelling.csv"
    TARGET_COL = 'Exited'
    DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

    # Load data
    X, y = load_data(DATA_PATH, TARGET_COL, drop_cols=DROP_COLS)

    # Split BEFORE SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Preprocessor 
    preprocessor = build_preprocessor(X)

    # Full pipeline: preprocessing -> smote -> model
    pipeline = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", get_model(model))
    ])

    # Train on training split
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save whole pipeline (recommended)
    save_model(pipeline, f"app/checkpoints/{model}.pkl")
    print(f"Model saved to app/checkpoints/{model}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "svm", "rf"],
        help="Model type"
    )

    args = parser.parse_args()
    main(args.model)