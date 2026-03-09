#!/usr/bin/env python3
"""
evaluate_model.py — Evaluate a trained model on test data.

Used in the DAGMan workflow (Episode 8) as the final evaluation step
after training completes.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn


class TitanicNet(nn.Module):
    """Same architecture as train_nn.py — must match for weight loading."""
    def __init__(self, d_in=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TitanicNet model")
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--data", required=True, help="Path to validation/test .npz")
    parser.add_argument("--output", default="evaluation.json",
                        help="Output evaluation results")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data)
    # Handle both train and val key names
    X = data.get("X_val", data.get("X_train")).astype("float32")
    y = data.get("y_val", data.get("y_train")).astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TitanicNet(d_in=X.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    # Evaluate
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).view(-1, 1).to(device)

    with torch.no_grad():
        preds = model(X_t)
        loss = nn.BCELoss()(preds, y_t).item()
        acc = (preds.round() == y_t).float().mean().item()

    results = {
        "model_path": args.model,
        "data_path": args.data,
        "num_samples": int(X.shape[0]),
        "loss": float(loss),
        "accuracy": float(acc),
        "device": str(device),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation Results:")
    print(f"  Samples:  {results['num_samples']}")
    print(f"  Loss:     {results['loss']:.6f}")
    print(f"  Accuracy: {results['accuracy']:.6f}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
