# Contrat de données

Les fichiers doivent être encodés en UTF-8, séparés par des virgules et contenir les en-têtes ci-dessous. Les dates utilisent le format ISO `YYYY-MM-DD`.

| Fichier | Colonnes obligatoires | Clé fonctionnelle |
| --- | --- | --- |
| `SalesData.csv` | `date`, `item_nbr`, `unit_sales` | `date`, `item_nbr` |
| `transactions.csv` | `date`, `transactions` | `date` |
| `oil.csv` | `date`, `oil_price` | `date` |
| `items.csv` | `item_nbr`, `family`, `class`, `perishable`, `has_float_sales` | `item_nbr` |
| `holidays.csv` | `date`, `is_Holiday`, `is_National`, `is_Local`, `is_likely_closed` | `date` |
| `PromotionCalendar.csv` | `date`, `item_nbr`, `onpromotion` | `date`, `item_nbr` |

## Règles

- `item_nbr` et `class` sont des entiers.
- `unit_sales`, `transactions` et `oil_price` sont numériques. Les ventes doivent être supérieures ou égales à zéro.
- Les indicateurs sont des booléens (`true`/`false`) ou des entiers `0`/`1`, sans valeurs manquantes.
- Les clés fonctionnelles ne doivent pas être dupliquées.
- Les données de ventes, de transactions et de pétrole doivent couvrir au moins 140 jours avant la date de prévision. Le calendrier des promotions et des jours fériés doit aussi couvrir les 28 jours suivants.

## Origine et diffusion

Conservez la source, la date d'extraction et les droits de redistribution pour chaque jeu de données. Si les données proviennent de Corporación Favorita/Kaggle ou d'une autre source sous licence, publiez le lien et les instructions de téléchargement au lieu de verser les CSV complets dans GitHub.
