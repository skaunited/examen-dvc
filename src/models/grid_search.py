import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle

# Chemins
PROCESSED_DATA_PATH = "data/processed_data"
MODELS_PATH = "models"

def main():
    # Charger les données
    X_train_scaled = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_train.csv").values.ravel()
    
    # Définir le modèle
    model = GradientBoostingRegressor(random_state=42)
    
    # Définir les paramètres à tester (grille réduite pour la rapidité)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    }
    
    # GridSearchCV
    print(f"Paramètres testés : {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Récupérer les meilleurs paramètres
    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres : {best_params}")
    print(f"Meilleur score (neg MSE) : {grid_search.best_score_:.4f}")
    
    with open(f"{MODELS_PATH}/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    print(f"Paramètres sauvegardés dans {MODELS_PATH}/best_params.pkl")

if __name__ == "__main__":
    main()