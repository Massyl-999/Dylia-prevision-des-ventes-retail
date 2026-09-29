# Données

`sample/` contient des exemples minimalistes qui documentent les schémas attendus. Les données réelles vont dans un dossier local ignoré par Git, par exemple `data/raw/` ou `Deployment_Data_Test/`.

L'application de production requiert au moins 140 jours d'historique par produit avant la date de prévision afin de calculer les variables de retard et les moyennes mobiles. Les échantillons ne sont donc pas destinés à entraîner le modèle ni à produire une prévision complète.

Consultez [../docs/DATA.md](../docs/DATA.md) pour les colonnes, les clés et les règles de validation.
