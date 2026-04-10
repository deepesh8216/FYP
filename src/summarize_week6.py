import argparse
import csv
import glob
import json
import os
import statistics


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Week 6 contrastive pretrain + attention fine-tune metrics"
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/week6/comparison_table.csv",
    )
    args = parser.parse_args()

    rows = []
    f1s = []
    accs = []
    for path in sorted(
        glob.glob("outputs/week6/finetune_attention/seed_*/test_metrics.json")
    ):
        seed = path.split("seed_")[-1].split("/")[0]
        with open(path, "r") as f:
            m = json.load(f)
        rows.append(
            {
                "model": "contrastive_then_attention",
                "seed": seed,
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "best_epoch": m.get("best_epoch"),
                "best_val_f1_macro": m.get("best_val_f1_macro"),
            }
        )
        f1s.append(m["f1_macro"])
        accs.append(m["accuracy"])

    if rows:
        rows.append(
            {
                "model": "contrastive_then_attention_mean",
                "seed": "all",
                "accuracy": sum(accs) / len(accs),
                "f1_macro": sum(f1s) / len(f1s),
                "precision_macro": "",
                "recall_macro": "",
                "best_epoch": "",
                "best_val_f1_macro": "",
            }
        )
        rows.append(
            {
                "model": "contrastive_then_attention_std",
                "seed": "all",
                "accuracy": statistics.pstdev(accs),
                "f1_macro": statistics.pstdev(f1s),
                "precision_macro": "",
                "recall_macro": "",
                "best_epoch": "",
                "best_val_f1_macro": "",
            }
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro",
                "best_epoch",
                "best_val_f1_macro",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
