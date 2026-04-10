"""
Week 7: merge experiment tables (Weeks 4–6) and emit ablation-friendly summaries.

Reads:
  outputs/week4/comparison_table.csv
  outputs/week5/comparison_table.csv
  outputs/week6/comparison_table.csv

Writes:
  outputs/week7/master_comparison.csv
  outputs/week7/ablation_summary.json
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path


WEEK4_MAP = {
    "text": "unimodal_text",
    "image": "unimodal_image",
    "fusion": "late_fusion",
}


def _is_numeric_seed(s):
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False


def load_week4_rows(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            variant = WEEK4_MAP.get(r.get("model", ""), r.get("model", ""))
            rows.append(
                {
                    "week": 4,
                    "variant": variant,
                    "seed": r["seed"],
                    "accuracy": float(r["accuracy"]),
                    "f1_macro": float(r["f1_macro"]),
                    "precision_macro": float(r["precision_macro"]),
                    "recall_macro": float(r["recall_macro"]),
                    "best_epoch": r.get("best_epoch") or "",
                    "best_val_f1_macro": r.get("best_val_f1_macro") or "",
                }
            )
    return rows


def load_week5_rows(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not _is_numeric_seed(r.get("seed")):
                continue
            rows.append(
                {
                    "week": 5,
                    "variant": "attention_fusion",
                    "seed": r["seed"],
                    "accuracy": float(r["accuracy"]),
                    "f1_macro": float(r["f1_macro"]),
                    "precision_macro": float(r["precision_macro"]),
                    "recall_macro": float(r["recall_macro"]),
                    "best_epoch": r.get("best_epoch") or "",
                    "best_val_f1_macro": r.get("best_val_f1_macro") or "",
                }
            )
    return rows


def load_week6_rows(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not _is_numeric_seed(r.get("seed")):
                continue
            rows.append(
                {
                    "week": 6,
                    "variant": "contrastive_then_attention",
                    "seed": r["seed"],
                    "accuracy": float(r["accuracy"]),
                    "f1_macro": float(r["f1_macro"]),
                    "precision_macro": float(r["precision_macro"]),
                    "recall_macro": float(r["recall_macro"]),
                    "best_epoch": r.get("best_epoch") or "",
                    "best_val_f1_macro": r.get("best_val_f1_macro") or "",
                }
            )
    return rows


def aggregate_by_variant(rows):
    by_v = {}
    for r in rows:
        key = (r["week"], r["variant"])
        by_v.setdefault(key, []).append(r)

    stats_rows = []
    for (week, variant), group in sorted(by_v.items()):
        f1s = [x["f1_macro"] for x in group]
        accs = [x["accuracy"] for x in group]
        stats_rows.append(
            {
                "week": week,
                "variant": variant,
                "seed": "mean",
                "accuracy": sum(accs) / len(accs),
                "f1_macro": sum(f1s) / len(f1s),
                "precision_macro": sum(x["precision_macro"] for x in group) / len(group),
                "recall_macro": sum(x["recall_macro"] for x in group) / len(group),
                "best_epoch": "",
                "best_val_f1_macro": "",
                "n_seeds": len(group),
            }
        )
        if len(group) > 1:
            stats_rows.append(
                {
                    "week": week,
                    "variant": variant,
                    "seed": "std",
                    "accuracy": statistics.pstdev(accs),
                    "f1_macro": statistics.pstdev(f1s),
                    "precision_macro": statistics.pstdev([x["precision_macro"] for x in group]),
                    "recall_macro": statistics.pstdev([x["recall_macro"] for x in group]),
                    "best_epoch": "",
                    "best_val_f1_macro": "",
                    "n_seeds": len(group),
                }
            )
    return stats_rows


def main():
    parser = argparse.ArgumentParser(description="Week 7: master table + ablation summary")
    parser.add_argument(
        "--week4_csv",
        default="outputs/week4/comparison_table.csv",
    )
    parser.add_argument(
        "--week5_csv",
        default="outputs/week5/comparison_table.csv",
    )
    parser.add_argument(
        "--week6_csv",
        default="outputs/week6/comparison_table.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/week7",
    )
    args = parser.parse_args()

    root = Path(".")
    w4 = root / args.week4_csv
    w5 = root / args.week5_csv
    w6 = root / args.week6_csv
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    per_seed.extend(load_week4_rows(w4))
    per_seed.extend(load_week5_rows(w5))
    per_seed.extend(load_week6_rows(w6))

    fieldnames = [
        "week",
        "variant",
        "seed",
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "best_epoch",
        "best_val_f1_macro",
    ]
    master_path = out_dir / "master_comparison.csv"
    with open(master_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(per_seed, key=lambda x: (x["week"], x["variant"], str(x["seed"]))):
            row = {k: r[k] for k in fieldnames}
            w.writerow(row)

    agg = aggregate_by_variant(per_seed)
    agg_path = out_dir / "aggregate_by_variant.csv"
    agg_fields = fieldnames + ["n_seeds"]
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for r in agg:
            w.writerow(r)

    means = [r for r in agg if r["seed"] == "mean"]
    ranking = sorted(
        means,
        key=lambda x: x["f1_macro"],
        reverse=True,
    )
    ablation = {
        "source_files": {
            "week4": str(w4),
            "week5": str(w5),
            "week6": str(w6),
        },
        "n_per_seed_rows": len(per_seed),
        "by_variant_mean": [
            {
                "week": r["week"],
                "variant": r["variant"],
                "mean_f1_macro": r["f1_macro"],
                "mean_accuracy": r["accuracy"],
                "n_seeds": r["n_seeds"],
            }
            for r in means
        ],
        "ranking_by_mean_f1_macro": [
            {
                "rank": i + 1,
                "week": r["week"],
                "variant": r["variant"],
                "mean_f1_macro": round(r["f1_macro"], 6),
                "mean_accuracy": round(r["accuracy"], 6),
            }
            for i, r in enumerate(ranking)
        ],
        "ablation_notes": [
            "unimodal_text / unimodal_image: modality removed from multimodal setting (Week 4 baselines).",
            "late_fusion: logits-level fusion without attention (Week 4).",
            "attention_fusion: cross-attention fusion without contrastive pretrain (Week 5).",
            "contrastive_then_attention: Week 6 pipeline (contrastive pretrain, then Week 5-style fine-tune).",
        ],
    }
    json_path = out_dir / "ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"Saved: {master_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {json_path}")
    print("\nRanking by mean F1 (macro):")
    for item in ablation["ranking_by_mean_f1_macro"][:10]:
        print(
            f"  {item['rank']}. W{item['week']} {item['variant']}: "
            f"F1={item['mean_f1_macro']:.4f}, Acc={item['mean_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
