"""
Download MulDar dataset from Hugging Face.

Usage:
    python evaluations/download_dataset.py                  # download all
    python evaluations/download_dataset.py --subset metal   # download one subset
    python evaluations/download_dataset.py --list           # list available subsets

Requires:
    pip install huggingface_hub
"""

import argparse
import os
from huggingface_hub import snapshot_download

REPO_ID = "xinghs/MulDar_Dataset"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..")

SUBSETS = [
    "metal",
    "plastic",
    "fabrics",
    "drywall",
    "wood",
    "curve",
    "deformable",
    "real_object",
    "car",
]


def download_dataset(subset=None):
    local_dir = os.path.abspath(LOCAL_DIR)
    os.makedirs(local_dir, exist_ok=True)

    if subset:
        print(f"Downloading subset '{subset}' to {local_dir}")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=f"adcData/{subset}/**",
        )
    else:
        print(f"Downloading full dataset to {local_dir}")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=local_dir,
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MulDar dataset from Hugging Face")
    parser.add_argument("--subset", type=str, default=None, choices=SUBSETS,
                        help="Download only a specific subset")
    parser.add_argument("--list", action="store_true", help="List available subsets")
    args = parser.parse_args()

    if args.list:
        print("Available subsets:")
        for s in SUBSETS:
            print(f"  - {s}")
    else:
        download_dataset(args.subset)
