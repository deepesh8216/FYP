import argparse
import csv
import glob
import json
import os


def collect_metrics(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        seed = path.split("seed_")[-1].split("/")[0] if "seed_" in path else "na"
        with open(path, "r") as f:
            metrics = json.load(f)
        rows.append(
            {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize Week 4 metrics")
    parser.add_argument("--output_csv", default="outputs/week4/comparison_table.csv")
    args = parser.parse_args()

    buckets = {
        "text": collect_metrics("outputs/week4/text/seed_*/test_metrics.json"),
        "image": collect_metrics("outputs/week4/image/seed_*/test_metrics.json"),
        "fusion": collect_metrics("outputs/week4/fusion/seed_*/test_metrics.json"),
    }

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "seed", "accuracy", "f1_macro", "precision_macro", "recall_macro"],
        )
        writer.writeheader()
        for model_name, rows in buckets.items():
            for row in rows:
                writer.writerow({"model": model_name, **row})

    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
