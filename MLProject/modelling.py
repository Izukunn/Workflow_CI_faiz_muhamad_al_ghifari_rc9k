import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import sys
import argparse
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(file_path):
    print(f"Mencoba membaca data dari: {file_path}")
    df = pd.read_csv(file_path)
    
    initial_shape = df.shape
    df = df.dropna()
    if df.shape != initial_shape:
        print(f"Warning: {initial_shape[0] - df.shape[0]} baris kosong dibuang.")
    
    X = df.drop('condition', axis=1)
    y = df['condition']
    return X, y

def train_with_tuning(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.csv")
    parser.add_argument("--test_data", type=str, default="data/test.csv")
    args = parser.parse_args()

    if "MLFLOW_TRACKING_URI" in os.environ:
        print("CI Environment detected. Menggunakan Env Vars.")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        print("Local Environment detected. Menjalankan dagshub.init()...")
        dagshub.init(repo_owner='Izukunn', 
                     repo_name='Eksperimen_SML_faiz_muhamad_al_ghifari_rc9k', 
                     mlflow=True)
    
    mlflow.set_experiment("Automated_CI_Experiment")

    try:
        X_train, y_train = load_data(args.train_data)
        X_test, y_test = load_data(args.test_data)
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan. {e}")
        sys.exit(1)

    with mlflow.start_run(run_name="CI_Run"):
        print("Mulai Training...")
        best_model, best_params = train_with_tuning(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, "model")
        
        plot_confusion_matrix(y_test, y_pred)
        plot_feature_importance(best_model, X_train.columns)
        
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("feature_importance.png")
        
        print("CI Run selesai.")