import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CreditDataPreprocessor:
    """Preprocesses the German Credit Risk dataset.

    Responsibilities
    - Scale numerical features (StandardScaler)
    - One-hot encode categorical features (OneHotEncoder)
    - Map target column "Risk" -> {"bad": 0, "good": 1}
    """

    def __init__(self) -> None:
        self.numerical_features = ["Age", "Job", "Credit amount", "Duration"]
        self.categorical_features = [
            "Sex",
            "Housing",
            "Saving accounts",
            "Checking account",
            "Purpose",
        ]
        self.target_feature = "Risk"

    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build and fit the preprocessing pipeline on the provided DataFrame.

        The DataFrame must contain the target column specified by ``self.target_feature``.
        """
        # Feature transformers
        numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_tf, self.numerical_features),
                ("cat", categorical_tf, self.categorical_features),
            ],
            remainder="passthrough",
        )

        # Fit on all features except the target
        x_train = df.drop(self.target_feature, axis=1)
        preprocessor.fit(x_train)
        return preprocessor

    def process_data(
        self, df: pd.DataFrame, preprocessor: ColumnTransformer
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply preprocessing and separate features/target.

        Args:
            df: DataFrame to process. Must contain the target column.
            preprocessor: Fitted ColumnTransformer.
        Returns:
            Tuple of (X_processed, y)
        """
        df_copy = df.copy()
        df_copy[self.target_feature] = df_copy[self.target_feature].map({"bad": 0, "good": 1})

        y = df_copy[self.target_feature]
        x = df_copy.drop(self.target_feature, axis=1)

        x_processed = preprocessor.transform(x)
        return x_processed, y
