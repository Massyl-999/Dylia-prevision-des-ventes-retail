# Liste de contrôle avant publication

- [ ] Créer un dépôt GitHub vide et définir son nom et sa visibilité.
- [ ] Vérifier qu'aucun fichier confidentiel, donnée client, secret ou clé API n'est présent avec `git status --ignored`.
- [ ] Laisser `Cache/`, les données complètes, les modèles `.pkl` et les études Optuna hors de Git.
- [ ] Vérifier les droits de redistribution des données et fournir un lien de téléchargement si elles ne peuvent pas être publiées.
- [x] Ajouter la licence MIT pour le code. La licence des données peut être différente.
- [ ] Mettre à jour le titre, l'URL du dépôt et l'auteur dans `README.md`.
- [ ] Installer `requirements.txt`, déposer les données et modèles localement, puis démarrer `python UI/app.py`.
- [ ] Confirmer que la page d'accueil et les routes `/api/kpi-summary` et `/api/predictions/details` répondent correctement.

## Première publication

```bash
git init
git add README.md requirements.txt .gitignore .gitattributes .env.example \
        Production.py UI data docs *.ipynb
git commit -m "Initial public release"
git branch -M main
git remote add origin <URL_DE_VOTRE_DEPOT>
git push -u origin main
```

Sous PowerShell, utilisez une ligne par commande ou remplacez les antislashs de continuation par des accents graves.
