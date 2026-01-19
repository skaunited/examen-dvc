import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_PATH = "data/processed_data"

def main():
    # Charger les données
    X_train = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_test.csv")
    
    # Normalisation avec StandardScaler
    scaler = StandardScaler()
    
    # Fit sur train, transform sur train et test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(f"{PROCESSED_DATA_PATH}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{PROCESSED_DATA_PATH}/X_test_scaled.csv", index=False)
    
    print(f"Données normalisées sauvegardées dans")
    print(f"  - X_train_scaled: {X_train_scaled.shape}")
    print(f"  - X_test_scaled: {X_test_scaled.shape}")

if __name__ == "__main__":
    main()