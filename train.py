import torch
import time
from pathlib import Path

from data import norm_stats, oneD_water_col, twoD_water_col, unnormalize_col
from model import build_edge_attr_dict

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

def train(model, dataloader, device=None, use_autocast=True, printevery=50):
    if device is None:
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    running_se_real = 0.0
    running_n = 0

    total_se_real = 0.0
    total_n = 0

    autocast_enabled = use_autocast and device.type == "mps"

    epoch_start = time.time()
    batch_times = []
    data_load_times = []

    dataiter = iter(dataloader)
    data_load_start = time.time()

    for i, data in enumerate(dataloader, start=1):
        data_load_time = time.time() - data_load_start
        data_load_times.append(data_load_time)
        compute_start = time.time()

        data = data.to(device)
        optimizer.zero_grad()

        edge_attr_dict = build_edge_attr_dict(data, model.edge_dim_std)

        if autocast_enabled:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)

                if "pred_mask" in data["oneD"]:
                    pred_oneDnodes_idxs = data["oneD"].pred_mask
                else:
                    pred_oneDnodes_idxs = data["oneD"].x[:, 0] == data["oneD"].x[:, 0].max()

                if "pred_mask" in data["twoD"]:
                    pred_twoDnodes_idxs = data["twoD"].pred_mask
                else:
                    pred_twoDnodes_idxs = data["twoD"].x[:, 0] == data["twoD"].x[:, 0].max()

                oneDpreds = out["oneD"][pred_oneDnodes_idxs].squeeze(-1)
                oneDtruth = data["oneD"].x[pred_oneDnodes_idxs, 2]

                twoDpreds = out["twoD"][pred_twoDnodes_idxs].squeeze(-1)
                twoDtruth = data["twoD"].x[pred_twoDnodes_idxs, 3]

                n1 = oneDpreds.numel()
                n2 = twoDpreds.numel()
                denom = max(n1 + n2, 1)

                loss1 = torch.mean((oneDpreds - oneDtruth) ** 2)
                loss2 = torch.mean((twoDpreds - twoDtruth) ** 2)

                loss = (loss1 * n1 + loss2 * n2) / denom
        else:
            out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)

            if "pred_mask" in data["oneD"]:
                pred_oneDnodes_idxs = data["oneD"].pred_mask
            else:
                pred_oneDnodes_idxs = data["oneD"].x[:, 0] == data["oneD"].x[:, 0].max()

            if "pred_mask" in data["twoD"]:
                pred_twoDnodes_idxs = data["twoD"].pred_mask
            else:
                pred_twoDnodes_idxs = data["twoD"].x[:, 0] == data["twoD"].x[:, 0].max()

            oneDpreds = out["oneD"][pred_oneDnodes_idxs].squeeze(-1)
            oneDtruth = data["oneD"].x[pred_oneDnodes_idxs, 2]

            twoDpreds = out["twoD"][pred_twoDnodes_idxs].squeeze(-1)
            twoDtruth = data["twoD"].x[pred_twoDnodes_idxs, 3]

            n1 = oneDpreds.numel()
            n2 = twoDpreds.numel()
            denom = max(n1 + n2, 1)

            loss1 = torch.mean((oneDpreds - oneDtruth) ** 2)
            loss2 = torch.mean((twoDpreds - twoDtruth) ** 2)

            loss = (loss1 * n1 + loss2 * n2) / denom

        loss.backward()
        optimizer.step()

        compute_time = time.time() - compute_start
        batch_time = data_load_time + compute_time
        batch_times.append(batch_time)

        # -------------------------
        # unnormalize for RMSE metric
        # -------------------------
        oneDpred_real = unnormalize_col(
            oneDpreds.detach(), norm_stats["oneD_mu"], norm_stats["oneD_sigma"], oneD_water_col
        )
        oneDtruth_real = unnormalize_col(
            oneDtruth.detach(), norm_stats["oneD_mu"], norm_stats["oneD_sigma"], oneD_water_col
        )

        twoDpred_real = unnormalize_col(
            twoDpreds.detach(), norm_stats["twoD_mu"], norm_stats["twoD_sigma"], twoD_water_col
        )
        twoDtruth_real = unnormalize_col(
            twoDtruth.detach(), norm_stats["twoD_mu"], norm_stats["twoD_sigma"], twoD_water_col
        )

        se_real = torch.sum((oneDpred_real - oneDtruth_real) ** 2)
        se_real += torch.sum((twoDpred_real - twoDtruth_real) ** 2)

        se_real = se_real.item()
        n_real = oneDpred_real.numel() + twoDpred_real.numel()

        running_se_real += se_real
        running_n += n_real

        total_se_real += se_real
        total_n += n_real

        # -------------------------
        # print intermediate TRUE RMSE
        # -------------------------
        if i % printevery == 0:
            rmse_real = (running_se_real / running_n) ** 0.5
            avg_batch_time = sum(batch_times[-printevery:]) / len(batch_times[-printevery:])
            avg_data_time = sum(data_load_times[-printevery:]) / len(data_load_times[-printevery:])
            print(f"Batch {i}: RMSE_real={rmse_real:.4f} | Avg time/batch={avg_batch_time:.3f}s (data={avg_data_time:.3f}s)")
            running_se_real = 0.0
            running_n = 0

        data_load_start = time.time()

    # -------------------------
    # final TRUE epoch RMSE
    # -------------------------
    epoch_rmse_real = (total_se_real / total_n) ** 0.5
    epoch_time = time.time() - epoch_start

    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_data_time = sum(data_load_times) / len(data_load_times) if data_load_times else 0
    print(
        f"  Epoch stats: {len(batch_times)} batches in {epoch_time:.1f}s | "
        f"Avg time/batch={avg_batch_time:.3f}s (data load={avg_data_time:.3f}s, compute={(avg_batch_time - avg_data_time):.3f}s)"
    )

    return epoch_rmse_real


def train_full(model, dataloader, epochs=2, device=None, use_autocast=True, printevery=50, checkpoint_dir="./checkpoints"):
    """Run full training loop over multiple epochs with checkpointing."""
    if device is None:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = model.to(device)
    model.train()

    score_train = []
    total_start = time.time()

    # Create checkpoint directory
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to {ckpt_path.resolve()}")

    for epoch in range(epochs):
        print(f"\nRunning Epoch: {epoch}")
        train_rmse = train(
            model,
            dataloader,
            device=device,
            use_autocast=use_autocast,
            printevery=printevery,
        )
        score_train.append(train_rmse)
        print(f"Epoch: {epoch:03d}, Train RMSE_real: {train_rmse:.4f}")

        # Save checkpoint after epoch
        ckpt_file = ckpt_path / f"epoch_{epoch:03d}_rmse_{train_rmse:.4f}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "rmse": train_rmse,
        }
        torch.save(checkpoint, ckpt_file)
        print(f"  Checkpoint saved: {ckpt_file.name}")

    model.eval()
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s for {epochs} epochs")
    print(f"Best RMSE: {min(score_train):.4f}")
    return model, score_train
