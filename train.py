import os
import torch
import time

from data import norm_stats, oneD_water_col, twoD_water_col, unnormalize_col, node1d_cols, node2d_cols
from model import build_edge_attr_dict

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

def _save_checkpoint(checkpoint_dir, tag, model, optimizer, epoch, batch_idx, rmse=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{tag}.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
    }
    if rmse is not None:
        payload["rmse"] = float(rmse)
    torch.save(payload, ckpt_path)


def train(model, dataloader, optimizer, device=None, use_autocast=False, printevery=50, checkpoint_dir=None, checkpoint_every=500, epoch_idx=0, global_step=0):
    if device is None:
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    if oneD_water_col is None or twoD_water_col is None:
        raise ValueError("water_level column not found for oneD/twoD; cannot compute accurate RMSE")

    reported_cols = set()

    def _check_tensor(name, t, col_names=None):
        if t is None:
            return True
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).sum().item()
            print(f"[WARN] Non-finite in {name}: {bad} values")
            if col_names is not None:
                bad_cols = (~torch.isfinite(t)).any(dim=0)
                idxs = bad_cols.nonzero(as_tuple=False).view(-1).tolist()
                if idxs:
                    key = f"{name}_cols"
                    if key not in reported_cols:
                        reported_cols.add(key)
                        names = [col_names[i] for i in idxs[:10]]
                        print(f"[WARN] Non-finite columns in {name}: {names}")
            return False
        max_abs = t.abs().max().item() if t.numel() > 0 else 0.0
        if max_abs > 1e6:
            print(f"[WARN] Large magnitude in {name}: max_abs={max_abs:.3e}")
        return True

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
        global_step += 1
        data_load_time = time.time() - data_load_start
        data_load_times.append(data_load_time)
        compute_start = time.time()

        data = data.to(device)
        optimizer.zero_grad()

        # Extract num_timesteps from batch (stored on node types to survive batching)
        if "num_timesteps" in data["oneD"]:
            nt = data["oneD"].num_timesteps
            num_timesteps = int(nt[0].item()) if nt.dim() > 0 else int(nt.item())
        elif "num_timesteps" in data["twoD"]:
            nt = data["twoD"].num_timesteps
            num_timesteps = int(nt[0].item()) if nt.dim() > 0 else int(nt.item())
        else:
            num_timesteps = None

        edge_attr_dict = build_edge_attr_dict(data, model.edge_dim_std)

        # Sanity check inputs before forward
        inputs_ok = True
        for ntype in data.node_types:
            cols = node1d_cols if ntype == "oneD" else node2d_cols if ntype == "twoD" else None
            inputs_ok = inputs_ok and _check_tensor(f"{ntype}.x", data[ntype].x, col_names=cols)
        for etype, ea in edge_attr_dict.items():
            inputs_ok = inputs_ok and _check_tensor(f"edge_attr{etype}", ea)
        if not inputs_ok:
            print("[WARN] Skipping batch due to non-finite inputs.")
            optimizer.zero_grad(set_to_none=True)
            data_load_start = time.time()
            continue

        if autocast_enabled:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                out = model(data.x_dict, data.edge_index_dict, edge_attr_dict, num_timesteps=num_timesteps, batch=data)

                if "pred_mask" in data["oneD"]:
                    pred_oneDnodes_idxs = data["oneD"].pred_mask
                else:
                    pred_oneDnodes_idxs = data["oneD"].x[:, 0] == data["oneD"].x[:, 0].max()

                if "pred_mask" in data["twoD"]:
                    pred_twoDnodes_idxs = data["twoD"].pred_mask
                else:
                    pred_twoDnodes_idxs = data["twoD"].x[:, 0] == data["twoD"].x[:, 0].max()

                oneDpreds = out["oneD"][pred_oneDnodes_idxs].squeeze(-1)
                oneDtruth = data["oneD"].y  # Ground truth already filtered to pred nodes

                twoDpreds = out["twoD"][pred_twoDnodes_idxs].squeeze(-1)
                twoDtruth = data["twoD"].y  # Ground truth already filtered to pred nodes

                n1 = oneDpreds.numel()
                n2 = twoDpreds.numel()
                denom = max(n1 + n2, 1)

                loss1 = torch.mean((oneDpreds - oneDtruth) ** 2)
                loss2 = torch.mean((twoDpreds - twoDtruth) ** 2)

                loss = (loss1 * n1 + loss2 * n2) / denom
        else:
            out = model(data.x_dict, data.edge_index_dict, edge_attr_dict, num_timesteps=num_timesteps, batch=data)

            if "pred_mask" in data["oneD"]:
                pred_oneDnodes_idxs = data["oneD"].pred_mask
            else:
                pred_oneDnodes_idxs = data["oneD"].x[:, 0] == data["oneD"].x[:, 0].max()

            if "pred_mask" in data["twoD"]:
                pred_twoDnodes_idxs = data["twoD"].pred_mask
            else:
                pred_twoDnodes_idxs = data["twoD"].x[:, 0] == data["twoD"].x[:, 0].max()

            oneDpreds = out["oneD"][pred_oneDnodes_idxs].squeeze(-1)
            oneDtruth = data["oneD"].y  # Ground truth already filtered to pred nodes

            twoDpreds = out["twoD"][pred_twoDnodes_idxs].squeeze(-1)
            twoDtruth = data["twoD"].y  # Ground truth already filtered to pred nodes

            n1 = oneDpreds.numel()
            n2 = twoDpreds.numel()
            denom = max(n1 + n2, 1)

            loss1 = torch.mean((oneDpreds - oneDtruth) ** 2)
            loss2 = torch.mean((twoDpreds - twoDtruth) ** 2)

            loss = (loss1 * n1 + loss2 * n2) / denom

        total_grad_norm = float("nan")

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if (not torch.isfinite(oneDpreds).all()) or (not torch.isfinite(twoDpreds).all()):
            print("[WARN] Non-finite predictions. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Compute per-layer gradient norms (diagnostic) - print every printevery batches
        if i % printevery == 0:
            layer_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    layer_key = ".".join(name.split(".")[:2]) if "." in name else name
                    if layer_key not in layer_grad_norms:
                        layer_grad_norms[layer_key] = []
                    layer_grad_norms[layer_key].append(grad_norm)

            if layer_grad_norms:
                avg_layer_norms = {k: sum(v) / len(v) for k, v in layer_grad_norms.items()}
                worst = sorted(avg_layer_norms.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"Top 3 layers by avg grad norm: {worst}")

        # Compute total grad norm without clipping
        total_grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_grad_sq += param_norm ** 2
        total_grad_norm = total_grad_sq ** 0.5
        if not torch.isfinite(torch.tensor(total_grad_norm)):
            print("[WARN] Non-finite grad norm. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        if checkpoint_dir is not None and checkpoint_every > 0 and global_step % checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_dir,
                tag=f"epoch{epoch_idx:03d}_batch{global_step}",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                batch_idx=global_step,
            )

        compute_time = time.time() - compute_start
        batch_time = data_load_time + compute_time
        batch_times.append(batch_time)

        # -------------------------
        # unnormalize for RMSE metric
        # -------------------------
        oneDpred_real = unnormalize_col(
            oneDpreds.detach(), norm_stats, oneD_water_col, 'oneD'
        )
        oneDtruth_real = unnormalize_col(
            oneDtruth.detach(), norm_stats, oneD_water_col, 'oneD'
        )

        twoDpred_real = unnormalize_col(
            twoDpreds.detach(), norm_stats, twoD_water_col, 'twoD'
        )
        twoDtruth_real = unnormalize_col(
            twoDtruth.detach(), norm_stats, twoD_water_col, 'twoD'
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
            print(f"Batch {i}: RMSE_real={rmse_real:.4f} | Loss_normalized={loss.item():.6f} | Grad_norm={total_grad_norm:.6f} | Avg time/batch={avg_batch_time:.3f}s (data={avg_data_time:.3f}s)")
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

    return epoch_rmse_real, global_step


def train_full(model, dataloader, epochs=2, device=None, use_autocast=False, printevery=50, checkpoint_dir="checkpoints", checkpoint_every=500):
    """Run full training loop over multiple epochs."""
    if device is None:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    score_train = []
    total_start = time.time()

    global_step = 0

    for epoch in range(epochs):
        print(f"\nRunning Epoch: {epoch}")
        train_rmse, global_step = train(
            model,
            dataloader,
            optimizer,
            device=device,
            use_autocast=use_autocast,
            printevery=printevery,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            epoch_idx=epoch,
            global_step=global_step,
        )
        score_train.append(train_rmse)
        print(f"Epoch: {epoch:03d}, Train RMSE_real: {train_rmse:.4f}")

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir,
                tag=f"epoch{epoch:03d}",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                batch_idx=global_step,
                rmse=train_rmse,
            )

    model.eval()
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s for {epochs} epochs")
    return model, score_train
