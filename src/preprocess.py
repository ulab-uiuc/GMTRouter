import argparse
from util import build_config


def main():
    parser = argparse.ArgumentParser(description="Pre process dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset Name")
    parser.add_argument("--base_dir", type=str, default="./data", required=False, help="Path to dataset")
    args = parser.parse_args()
    base_dir = args.base_dir
    dataset = args.dataset
    build_config(f"{base_dir}/{dataset}/training_set.jsonl", ckpt_path=f"{base_dir}/{dataset}/training_set.pt")

if __name__ == "__main__":
    main()