import torch

from data import get_dataloader, edge_col_info, oneD_water_col, twoD_water_col, norm_stats
from model import build_model_from_batch
from train import train_full


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dl = get_dataloader()
    batch0 = next(iter(dl))
    # Build model with learned predictions (not mass conservation)
    model = build_model_from_batch(
        batch0, 1,
        edge_col_info=edge_col_info,
        water_level_indices={'oneD': oneD_water_col, 'twoD': twoD_water_col},
        use_cons_of_mass=False,  # Disable mass conservation - use learned model
        norm_stats=norm_stats,
    )

    model, scores = train_full(model, dl, epochs=2, device=device, use_autocast=False, printevery=100)
    print(f"\nTraining complete. Final RMSE: {scores[-1]:.4f}")


if __name__ == "__main__":
    main()

