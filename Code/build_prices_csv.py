import os
import pandas as pd

# 📁 Dossier contenant les fichiers CSV de chaque actif
data_folder = "Data/"
files = ["BNP.csv", "Airbus.csv", "Deutsche.csv", "Enel.csv", "LVMH.csv", "Sanofi.csv"]

all_prices = {}

for file in files:
    ticker = file.replace(".csv", "")
    path = os.path.join(data_folder, file)

    # Chargement des données
    df = pd.read_csv(path, parse_dates=True)
    
    # Vérification du nom de la colonne "Date" si nécessaire
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Stocker la série des prix "Close"
    all_prices[ticker] = df["Close"]

# 🔁 Combinaison des séries et suppression des dates incomplètes
prices_df = pd.DataFrame(all_prices).dropna()

# 💾 Sauvegarde du fichier CSV
output_path = "Data/prices_aligned.csv"
prices_df.to_csv(output_path)

print(f"✅ Fichier sauvegardé : {output_path}")
print(prices_df.head())
