# Dylia — prévision des ventes retail

Application de prévision à 28 jours des ventes par produit. Le projet prépare les données de vente, crée des variables temporelles et de promotion, puis sert les prédictions via une API Flask et une interface web.

## Ce que contient le dépôt

- `Production.py` : pipeline de préparation, création des variables et inférence LightGBM.
- `UI/` : API Flask et interface statique.
- `analyze.ipynb` : exploration et analyse des données.
- `features_engineering.ipynb` : expérimentation de l'ingénierie des variables.
- `Model_Training*.ipynb` : entraînement LightGBM et XGBoost.
- `HyperParameters_Tuning*.ipynb` : optimisation Optuna, dont une variante GPU pour XGBoost.
- `data/sample/` : petits CSV versionnés qui documentent les formats d'entrée.
- `data/raw/` : emplacement local des données réelles, volontairement ignoré par Git.
- `docs/` : schémas des données et liste de contrôle avant publication.

## Installation

Prérequis : Python 3.10 ou plus récent.

```bash
git clone <URL_DE_VOTRE_DEPOT>
cd Dylia_Project
python -m venv .venv
```

Sous Windows :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python UI\app.py
```

L'interface est ensuite disponible sur `http://localhost:5000`.

## Données et modèles

Les données complètes, les caches et les modèles entraînés ne sont pas versionnés. Ils sont volumineux et peuvent être soumis à des contraintes de licence. Avant de lancer l'application, placez les six fichiers décrits dans [docs/DATA.md](docs/DATA.md) dans le dossier configuré par `DYLIA_DATA_DIR` (par défaut : `Deployment_Data_Test/`).

Placez les 28 artefacts `Model1.pkl` à `Model28.pkl` dans `DYLIA_MODELS_DIR` (par défaut : `Models/`). Les caches sont régénérés dans `DYLIA_CACHE_DIR`.

Les CSV dans `data/sample/` sont des exemples de schéma et ne contiennent pas assez d'historique pour exécuter une prévision complète.

## Variables d'environnement

Copiez `.env.example` vers `.env`, puis adaptez les chemins si nécessaire :

```dotenv
DYLIA_DATA_DIR=Deployment_Data_Test
DYLIA_CACHE_DIR=Cache
DYLIA_MODELS_DIR=Models
```

Les chemins relatifs sont interprétés depuis la racine du projet. Des chemins absolus sont également acceptés.

Pour vérifier les six CSV avant le lancement :

```bash
python scripts/validate_input_data.py --data-dir Deployment_Data_Test
```

## Publication GitHub

Ne publiez pas les répertoires `Deployment_Data_Test/`, `Store39_Dataset/`, `Cache/`, `Models/`, `Models_XGB/` ni les études Optuna. Ils sont déjà protégés par `.gitignore`.

Suivez [docs/PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) avant le premier `git push`. Vérifiez en particulier les droits de redistribution des données.

## Licence

Le code est distribué sous licence [MIT](LICENSE). Cette licence s'applique au code du dépôt, pas automatiquement aux jeux de données externes ni aux modèles entraînés. Vérifiez et documentez les droits de redistribution de ces artefacts séparément.
