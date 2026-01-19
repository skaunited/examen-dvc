import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Chemins
PROCESSED_DATA_PATH = "data/processed_data"
MODELS_PATH = "models"

def main():
    # Charger les données
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_train.csv").values.ravel()
    
    # Charger les meilleurs paramètres
    with open(f"{MODELS_PATH}/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    
    print(f"Paramètres utilisés : {best_params}")
    
    # Créer et entraîner le modèle
    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train_scaled, y_train)
    
    # Sauvegarder le modèle
    with open(f"{MODELS_PATH}/gbr_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(f"Modèle sauvegardé dans {MODELS_PATH}/gbr_model.pkl")

if __name__ == "__main__":
    main()