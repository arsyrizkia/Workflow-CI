"""
modelling.py (versi MLflow Project / CI)
========================================
Dijalankan oleh `mlflow run` dari workflow CI (GitHub Actions) untuk melakukan
re-training model setiap kali trigger terpantik.

Output:
- Run MLflow (parameter, metrik, artefak model) pada tracking store lokal runner;
- Salinan model pada folder `artifacts/model` (dipakai `mlflow models build-docker`);
- File `run_id.txt` berisi run id untuk langkah workflow berikutnya.
"""

import argparse
import os
import shutil

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TARGET_COL = "HeartDisease"
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="heart_disease_preprocessing/heart_preprocessed.csv")
    parser.add_argument("--n_estimators", type=float, default=300)
    parser.add_argument("--max_depth", type=float, default=10)
    args = parser.parse_args()

    n_estimators = int(args.n_estimators)
    max_depth = None if args.max_depth <= 0 else int(args.max_depth)

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="ci_retraining") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Salinan model untuk langkah build-docker & penyimpanan artefak di repo
        model_dir = os.path.join("artifacts", "model")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(model, model_dir)

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"run_id        : {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name:<15}: {value:.4f}")


if __name__ == "__main__":
    main()
