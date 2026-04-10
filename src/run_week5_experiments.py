import argparse
import subprocess


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run Week 5 attention-fusion experiments")
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        out = f"outputs/week5/attention/seed_{seed}"
        run(f"python3 src/train_attention_fusion.py --seed {seed} --epochs {args.epochs} --output_dir {out}")
        run(
            "python3 src/evaluate_attention_fusion.py "
            f"--checkpoint {out}/best_model.pt "
            f"--output_json {out}/test_metrics.json"
        )

    print("\nWeek 5 runs completed.")


if __name__ == "__main__":
    main()
