"""Download model weights and dataset on the login node, so GPU jobs never wait on the network.

    python scripts/prefetch.py                       # default model
    python scripts/prefetch.py --model Qwen/Qwen3-4B
"""

import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download

from longtail.config import RolloutConfig

if __name__ == "__main__":
    d = RolloutConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.model)
    args = ap.parse_args()
    print("model ->", snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
    ds = load_dataset(d.dataset, d.dataset_config, split=d.split)
    print(f"dataset -> {len(ds)} rows")
