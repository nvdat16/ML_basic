import argparse
import pandas as pd
from utils import load_model


def main(model, input_path):
    checkpoint = load_model(model)
    model = checkpoint["model"]
    preprocessor = checkpoint["preprocessor"]

    X_new = pd.read_csv(input_path)
    X_new_processed = preprocessor.transform(X_new)

    predictions = model.predict(X_new_processed)
    print("Predictions:", predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model type"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input CSV file"
    )

    args = parser.parse_args()
    main(args.model, args.input_path)