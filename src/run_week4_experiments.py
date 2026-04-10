import argparse
import subprocess


def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run week 4 experiments for multiple seeds")
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        run(
            f"python3 src/train_text.py --seed {seed} --epochs {args.epochs} "
            f"--output_dir outputs/week4/text/seed_{seed}"
        )
        run(
            f"python3 src/evaluate_text.py "
            f"--checkpoint outputs/week4/text/seed_{seed}/text_model.pt "
            f"--output_json outputs/week4/text/seed_{seed}/test_metrics.json"
        )

        run(
            f"python3 src/train_image.py --seed {seed} --epochs {args.epochs} "
            f"--output_dir outputs/week4/image/seed_{seed}"
        )
        run(
            f"python3 src/evaluate_image.py "
            f"--checkpoint outputs/week4/image/seed_{seed}/image_model.pt "
            f"--output_json outputs/week4/image/seed_{seed}/test_metrics.json"
        )

        run(
            f"python3 src/train_late_fusion.py --seed {seed} --epochs {args.epochs} "
            f"--output_dir outputs/week4/fusion/seed_{seed}"
        )
        run(
            f"python3 src/tune_fusion_alpha.py "
            f"--checkpoint outputs/week4/fusion/seed_{seed}/fusion_model.pt "
            f"--output_json outputs/week4/fusion/seed_{seed}/fusion_alpha_search.json"
        )
        run(
            f"python3 src/evaluate_late_fusion.py "
            f"--checkpoint outputs/week4/fusion/seed_{seed}/fusion_model.pt "
            f"--output_json outputs/week4/fusion/seed_{seed}/test_metrics.json"
        )

    print("\nWeek 4 runs completed.")


if __name__ == "__main__":
    main()
