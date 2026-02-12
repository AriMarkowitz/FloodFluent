import torch

from data import dl
from model import build_model_from_batch
from train import train_full


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    batch0 = next(iter(dl))
    model = build_model_from_batch(batch0, 1)

    model, scores = train_full(model, dl, epochs=2, device=device, use_autocast=True, printevery =5)
    print(f"\nTraining complete. Final RMSE: {scores[-1]:.4f}")


if __name__ == "__main__":
    main()

