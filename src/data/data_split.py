import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw_data/raw.csv"
PROCESSED_DATA_PATH = "data/processed_data"

FEATURE_COLUMNS = [
    "ave_flot_air_flow",
    "ave_flot_level",
    "iron_feed",
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_pH",
    "ore_pulp_density"
]
TARGET_COLUMN = "silica_concentrate"

def main():
    # Charger les données
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"Colonnes disponibles : {df.columns.tolist()}")
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    print("On split des données 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
        
    # Sauvegarder les datasets
    X_train.to_csv(f"{PROCESSED_DATA_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_PATH}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_PATH}/y_test.csv", index=False)
    
    print(f"Données sauvegardées")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")

if __name__ == "__main__":
    main()