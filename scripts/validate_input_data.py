"""Validate the input CSV schemas required by the Dylia forecasting pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "SalesData.csv": ("date", "item_nbr", "unit_sales"),
    "transactions.csv": ("date", "transactions"),
    "oil.csv": ("date", "oil_price"),
    "items.csv": ("item_nbr", "family", "class", "perishable", "has_float_sales"),
    "holidays.csv": ("date", "is_Holiday", "is_National", "is_Local", "is_likely_closed"),
    "PromotionCalendar.csv": ("date", "item_nbr", "onpromotion"),
}

KEY_COLUMNS = {
    "SalesData.csv": ("date", "item_nbr"),
    "transactions.csv": ("date",),
    "oil.csv": ("date",),
    "items.csv": ("item_nbr",),
    "holidays.csv": ("date",),
    "PromotionCalendar.csv": ("date", "item_nbr"),
}


def validate_csv(path: Path, required_columns: tuple[str, ...], key_columns: tuple[str, ...]) -> list[str]:
    """Return a list of schema and duplicate-key issues for one CSV file."""
    issues: list[str] = []
    if not path.is_file():
        return ["fichier introuvable"]

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        missing = [column for column in required_columns if column not in headers]
        if missing:
            return [f"colonnes manquantes : {', '.join(missing)}"]

        seen_keys: set[tuple[str | None, ...]] = set()
        for line_number, row in enumerate(reader, start=2):
            key = tuple(row[column] for column in key_columns)
            if any(value in (None, "") for value in key):
                issues.append(f"ligne {line_number} : clé vide ({', '.join(key_columns)})")
            elif key in seen_keys:
                issues.append(f"ligne {line_number} : clé dupliquée {key}")
            seen_keys.add(key)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Valide les fichiers CSV d'entrée de Dylia.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dossier contenant les six CSV.")
    args = parser.parse_args()

    has_issues = False
    for filename, columns in REQUIRED_COLUMNS.items():
        issues = validate_csv(args.data_dir / filename, columns, KEY_COLUMNS[filename])
        if issues:
            has_issues = True
            for issue in issues:
                print(f"ERREUR {filename} : {issue}")
        else:
            print(f"OK {filename}")

    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
