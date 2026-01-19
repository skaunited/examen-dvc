import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import numpy as np

PROCESSED_DATA_PATH = "data/processed_data"
MODELS_PATH = "models"
METRICS_PATH = "metrics"

def main():
    # Charger les données de test
    X_test_scaled = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_test.csv").values.ravel()
    
    # Charger le modèle
    with open(f"{MODELS_PATH}/gbr_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Faire des prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Métriques d'évaluation :")
    print(f"  - MSE  : {mse:.4f}")
    print(f"  - RMSE : {rmse:.4f}")
    print(f"  - MAE  : {mae:.4f}")
    print(f"  - R2   : {r2:.4f}")
    
    # Sauvegarder les prédictions
    predictions_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred
    })
    predictions_df.to_csv(f"{PROCESSED_DATA_PATH}/predictions.csv", index=False)
    print(f"Prédictions sauvegardées dans {PROCESSED_DATA_PATH}/predictions.csv")
    
    scores = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    with open(f"{METRICS_PATH}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    print(f"Métriques sauvegardées dans {METRICS_PATH}/scores.json")

if __name__ == "__main__":
    main()