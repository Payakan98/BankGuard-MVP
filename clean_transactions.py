import pandas as pd

INPUT = "data/transactions_sample.csv"
OUTPUT = "data/transactions_sample_clean.csv"

def clean_csv(input_path=INPUT, output_path=OUTPUT):
    # Lire le CSV sans parsing automatique des dates
    df = pd.read_csv(input_path, dtype=str)  # tout en str pour éviter conversion automatique

    # Convertir la colonne timestamp en datetime (tolérant aux erreurs)
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Afficher le nombre de lignes invalides
    n_before = len(df)
    n_bad = df['timestamp_parsed'].isna().sum()
    print(f"Lignes totales : {n_before}, timestamps invalides : {n_bad}")

    # Supprimer les lignes avec timestamp invalides
    df_clean = df[df['timestamp_parsed'].notna()].copy()

    # (Optionnel) — réinitialiser l’index
    df_clean.reset_index(drop=True, inplace=True)

    # Enlever la colonne temporaire ou la renommer
    df_clean.drop(columns=['timestamp'], inplace=True)
    df_clean = df_clean.rename(columns={'timestamp_parsed': 'timestamp'})

    # Sauver le CSV nettoyé
    df_clean.to_csv(output_path, index=False)
    print(f"CSV nettoyé sauvegardé dans: {output_path}")

if __name__ == "__main__":
    clean_csv()
