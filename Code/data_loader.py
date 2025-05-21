import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Charge un fichier CSV contenant les prix journaliers avec 'Date' comme index.
    - Ignore la colonne 'Change %' si présente.
    - Convertit le volume en float (ex: '2.89M' → 2_890_000).
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")

    # Supprimer "Change %" si présente
    if "Change %" in df.columns:
        df.drop(columns=["Change %"], inplace=True)


    df = df.sort_index()
    return df

def compute_log_returns(price_series):
    """
    Calcule les rendements log à partir d'une série de prix (Close).
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    return log_returns
