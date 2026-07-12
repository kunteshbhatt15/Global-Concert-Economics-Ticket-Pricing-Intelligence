from __future__ import annotations

import argparse
from pathlib import Path

from model_pipeline import fit_artifacts, load_dataset, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/concert_dataset_cleaned.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    df = load_dataset(args.data)
    artifacts = fit_artifacts(df)
    save_artifacts(artifacts, args.models)
    print(f"Saved tuned XGBoost artifacts to {Path(args.models).resolve()}")
    for metric, value in artifacts["metrics"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
