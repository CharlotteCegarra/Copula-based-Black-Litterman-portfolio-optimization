# Copula-Based Black–Litterman Portfolio Optimization

## Description

Ce projet a pour objectif de construire une stratégie d’**optimisation de portefeuille** en combinant :

- des **modèles GARCH** pour capturer la volatilité conditionnelle des actifs,
- des **copules (Vine, Student-t, Clayton)** pour modéliser les dépendances multivariées entre actifs financiers,
- un modèle **Black–Litterman enrichi par des prévisions VECM** pour intégrer des vues directionnelles sur les marchés,
- des **critères d’optimisation** tels que le ratio de Sharpe, la CVaR (Conditional Value at Risk) et le ratio STARR.

L’approche repose sur la simulation de rendements conditionnels, puis l’optimisation dynamique des portefeuilles simulés.

---

## Prérequis

Pour utiliser ce projet, vous devez d'abord créer un environnement virtuel et installer les packages répertoriés dans le fichier `requirements.txt`.

## Installation

1. Cloner le dépôt sur votre machine :
   ```bash
   git clone https://github.com/votre-utilisateur/Copula-based-Black-Litterman-portfolio-optimization.git
   cd Copula-based-Black-Litterman-portfolio-optimization

2. Créer un environnement virtuel
   ```bash
   python -m venv .venv

3. Activer l’environnement virtuel :
   ```bash
   .venv\Scripts\activate   # sous Windows
   source .venv/bin/activate   # sous macOS/Linux
 
4. Installer les dépendances avec pip
   ```bash
   pip install -r requirements.txt

5. Executer le script principal
   ```bash
   python main.py

Ce script :
- Charge les données de prix,
- Estime les modèles GARCH,
- Génère des pseudo-observations via copules,
- Produit des prévisions VECM,
- Calcule les rendements Black–Litterman simulés,
- Optimise les portefeuilles,
- Génère des visualisations et ouvre les rapports PDF.

## Structure des fichiers
```
/main.py                        # Script principal
/README.md                      # Documentation du projet
/requirements.txt               # Dépendances Python
/Sahamkhadam Stephan Ostermark  # Papier
/Rapport                        # Rapport du projet

/Code                        # Scripts utilisés dans le main
   ├── __init__.py
   ├── data_loader.py            # Chargement des prix et rendements log
   ├── garch_models.py           # Modèles GARCH + standardisation
   ├── copula_models.py          # Copules bivariées et Vine
   ├── vecm_views.py             # VECM + vues Black–Litterman
   ├── black_litterman.py        # BL : équilibre, vues, postérieur
   ├── optimization.py           # Fonctions d’optimisation
   └── build_prices_csv.py       # (optionnel) script pour générer les CSV

/Data                        # Données d'entrée (actions historiques)
   ├── Airbus.csv
   ├── BNP.csv
   ├── Deutsche.csv
   ├── Enel.csv
   ├── LVMH.csv
   ├── Sanofi.csv
   └── prices_aligned.csv        # Pour le VECM

/Output                       # Résultats/ Graphiques générés 
   ├── log_returns_all.csv
   ├── pseudo_observations.csv
   ├── weights_optimisés.csv
   ├── Rapport_Visualisations.pdf
   ├── Black-Litterman/
   ├── GARCH/
   ├── Optimisation PF/
   └── Rendement Log/
```
## Contributeurs
- **Charlotte CEGARRA**
- **Salma BENMOUSSA**
- **Chirine DEXPOSITO**
- **Hella BOUHADDA**
  
Ce projet a été développé dans le cadre du Master MOSEF, à l'université Paris 1 Panthéon Sorbonne.

## 📩 Contact

N'hésitez pas à nous contacter pour toute question :

- charlottecegarrapro@gmail.com
- salmabenmoussa103@gmail.com
- chirinedexposito@gmail.com
- lalabou0@gmail.com
