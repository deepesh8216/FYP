"""
Week 7: export confusion-matrix heatmaps from saved test_metrics.json files.
"""

import argparse
import json
import os
import re
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print(
        "week7_export_figures: matplotlib/seaborn not installed; "
        "skipping PNG export. Install with: pip install matplotlib seaborn"
    )
    raise SystemExit(0)


DEFAULT_GLOBS = [
    "outputs/week4/text/seed_*/test_metrics.json",
    "outputs/week4/image/seed_*/test_metrics.json",
    "outputs/week4/fusion/seed_*/test_metrics.json",
    "outputs/week5/attention/seed_*/test_metrics.json",
    "outputs/week6/finetune_attention/seed_*/test_metrics.json",
]


def slug_from_path(path: Path) -> str:
    parts = path.parts
    try:
        i = parts.index("outputs")
        tail = parts[i + 1 :]
    except ValueError:
        tail = path.parts
    s = "_".join(tail[:-1]).replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_") or "metrics"


def plot_confusion(cm, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export Week 7 confusion matrix PNGs")
    parser.add_argument(
        "--out_dir",
        default="outputs/week7/figures",
    )
    parser.add_argument(
        "--globs",
        nargs="*",
        default=None,
        help="Override default glob list",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    globs = args.globs if args.globs else DEFAULT_GLOBS
    seen = set()
    n = 0
    for pattern in globs:
        for path in sorted(Path(".").glob(pattern)):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            with open(path) as f:
                data = json.load(f)
            cm = data.get("confusion_matrix")
            if not cm or len(cm) != 2:
                continue
            slug = slug_from_path(Path(path))
            title = slug.replace("_", " ")
            plot_confusion(cm, title, out_root / f"{slug}_confusion.png")
            n += 1

    manifest = {"n_plots": n, "out_dir": str(out_root), "globs": globs}
    manifest_path = Path("outputs/week7/figures_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {n} PNGs under {out_root}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
