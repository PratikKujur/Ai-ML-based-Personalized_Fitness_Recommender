import kagglehub
import pandas as pd
import os
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import joblib

from src.constants import model_params
from src.utils import (
    preprocess_for_xgb,
    preprocess_categorical,
    preprocess_target,
    dropna_func,
)

data_path = os.path.join(os.getcwd(), "dataset")


class pipline:
    def __init__(self):
        pass

    def get_data(self):
        os.curdir
        path = kagglehub.dataset_download("jockeroika/life-style-data")
        os.makedirs(data_path, exist_ok=True)
        os.system(f'copy "{path}" "{data_path}"')

    def data_transforma(self):
        df = pd.read_csv(os.path.join(data_path, "Final_data.csv"))
        # feature selection is based on domain knowledge
        cat_features = [
            "Gender",
            "diet_type",
            "cooking_method",
            "Equipment Needed",
            "Difficulty Level",
            "Body Part",
        ]
        num_features = [
            "Age",
            "Water_Intake (liters)",
            "Height (m)",
            "Weight (kg)",
            "Session_Duration (hours)",
            "Workout_Frequency (days/week)",
            "Experience_Level",
            "Daily meals frequency",
        ]
        target_feature = "Name of Exercise"
        all_features = cat_features + num_features + [target_feature]
        all_features.sort()
        df_filtered = df[all_features]
        df_filtered = preprocess_target(df_filtered, target_feature)
        df_filtered = preprocess_categorical(df_filtered, cat_features)
        return df_filtered, target_feature

    def model_trainer(self, df, target):
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        xgb_model = xgb.XGBClassifier(**model_params)
        preprocess_transformer = FunctionTransformer(preprocess_for_xgb, validate=False)
        pipeline = Pipeline(
            steps=[
                ("dropna", FunctionTransformer(dropna_func, validate=False)),
                ("preprocess", preprocess_transformer),
                ("model", xgb_model),
            ]
        )
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, "models\exercise_recommender.pkl")

    def run_pipeline(self):
        self.get_data()
        df, target = self.data_transforma()
        self.model_trainer(df, target)


if __name__ == "__main__":
    model_pipeline = pipline()
    model_pipeline.run_pipeline()
