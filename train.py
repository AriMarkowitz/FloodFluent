import torch

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

def train(model, dataloader, printevery=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    running_se_real = 0.0
    running_n = 0

    total_se_real = 0.0
    total_n = 0

    for i, data in enumerate(dataloader, start=1):

        data = data.to(device)
        optimizer.zero_grad()

        edge_attr_dict = build_edge_attr_dict(data, model.edge_dim_std)

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

        # -------------------------
        # normalized weighted loss
        # -------------------------
        n1 = oneDpreds.numel()
        n2 = twoDpreds.numel()
        denom = max(n1 + n2, 1)

        loss1 = torch.mean((oneDpreds - oneDtruth)**2)
        loss2 = torch.mean((twoDpreds - twoDtruth)**2)

        loss = (loss1 * n1 + loss2 * n2) / denom
        loss.backward()
        optimizer.step()

        # -------------------------
        # unnormalize for RMSE metric
        # -------------------------
        oneDpred_real = unnormalize_col(
            oneDpreds, norm_stats["oneD_mu"], norm_stats["oneD_sigma"], oneD_water_col
        )
        oneDtruth_real = unnormalize_col(
            oneDtruth, norm_stats["oneD_mu"], norm_stats["oneD_sigma"], oneD_water_col
        )

        twoDpred_real = unnormalize_col(
            twoDpreds, norm_stats["twoD_mu"], norm_stats["twoD_sigma"], twoD_water_col
        )
        twoDtruth_real = unnormalize_col(
            twoDtruth, norm_stats["twoD_mu"], norm_stats["twoD_sigma"], twoD_water_col
        )

        se_real = torch.sum((oneDpred_real - oneDtruth_real)**2)
        se_real += torch.sum((twoDpred_real - twoDtruth_real)**2)

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
            print(f"Batch {i}: RMSE_real={rmse_real:.4f}")
            running_se_real = 0.0
            running_n = 0

    # -------------------------
    # final TRUE epoch RMSE
    # -------------------------
    epoch_rmse_real = (total_se_real / total_n) ** 0.5
    return epoch_rmse_real


score_train = []
epochs = 2
model = model.to(device)
model.train()

for epoch in range(epochs):
    print("Running Epoch:", epoch)
    train_rmse = train(model, dl)
    score_train.append(train_rmse)
    print(f'Epoch: {epoch:03d}, Train RMSE_real: {train_rmse:.4f}')
model.eval()