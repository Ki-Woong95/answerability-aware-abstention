"""Dataset-only entry point to the preserved all-layer training algorithm."""

import argparse
import torch
from src.gate.datasets import get_dataset_config
from src.gate.train_all_layers import train_model, EPOCHS


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, choices=["af_next", "moss", "audio-reasoner"]
    )
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    train_model(
        args.model,
        torch.device(args.device),
        EPOCHS,
        None,
        False,
        get_dataset_config("treecut"),
    )


if __name__ == "__main__":
    main()
