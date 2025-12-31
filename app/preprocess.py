import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(data_path, target_col, drop_cols=None):
    df = pd.read_csv(data_path)

    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y



def build_preprocessor(X):
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
