"""
Week 6: contrastive pretraining, then attention-fusion fine-tune (same architecture as Week 5).

Writes per seed:
  outputs/week6/contrastive/seed_<seed>/contrastive_pretrained.pt
  outputs/week6/finetune_attention/seed_<seed>/best_model.pt
  outputs/week6/finetune_attention/seed_<seed>/test_metrics.json

Then: python3 src/summarize_week6.py
"""

import argparse
import subprocess


def run(cmd: str) -> None:
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Week 6 contrastive pretrain + attention finetune for multiple seeds"
    )
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=8)
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        contrastive_dir = f"outputs/week6/contrastive/seed_{seed}"
        finetune_dir = f"outputs/week6/finetune_attention/seed_{seed}"
        run(
            f"python3 src/pretrain_contrastive.py --seed {seed} "
            f"--epochs {args.pretrain_epochs} --output_dir {contrastive_dir}"
        )
        run(
            f"python3 src/train_attention_fusion.py --seed {seed} "
            f"--epochs {args.finetune_epochs} --output_dir {finetune_dir} "
            f"--init_contrastive {contrastive_dir}/contrastive_pretrained.pt"
        )
        run(
            "python3 src/evaluate_attention_fusion.py "
            f"--checkpoint {finetune_dir}/best_model.pt "
            f"--output_json {finetune_dir}/test_metrics.json"
        )

    print("\nWeek 6 runs completed. Summarize: python3 src/summarize_week6.py")


if __name__ == "__main__":
    main()
