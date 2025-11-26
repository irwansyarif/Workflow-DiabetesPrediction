import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # ============================================
    # FIX PATH CSV → selalu aman di Jupyter & Terminal & MLflow
    # ============================================
    default_csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../MLProject/diabetes_prediction_dataset_train.csv"
    )

    # Ambil argumen ke-1 kalau ada, kalau tidak maka pakai default
    file_path = default_csv_path
    for arg in sys.argv:
        if arg.endswith(".csv"):
            file_path = arg

    print("Using dataset:", file_path)
    data = pd.read_csv(file_path)

    # ============================================
    # FIX SYS.ARGV → hanya menerima angka jika valid
    # ============================================
    def get_int_arg(index, default):
        try:
            return int(sys.argv[index])
        except:
            return default

    n_estimators = get_int_arg(1, 505)
    max_depth   = get_int_arg(2, 37)
 
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop("diabetes", axis=1),
    data["diabetes"],
    random_state=42,
    test_size=0.2
    )
    input_example = X_train[0:5]
 
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )