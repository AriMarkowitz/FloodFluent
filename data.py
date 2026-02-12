import os
import pandas as pd
import torch
import torch.utils.data as tud
from torch.utils.data import IterableDataset, ChainDataset
from torch_geometric.loader import DataLoader
import torch_geometric as tg
import glob
from pathlib import Path

# =========================
# Running stats (Welford)
# =========================
class RunningMeanStd:
    def __init__(self, n_features: int):
        self.n = 0
        self.mean = torch.zeros(n_features, dtype=torch.float64)
        self.M2 = torch.zeros(n_features, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        # x: [N, F]
        x = x.to(torch.float64)
        batch_n = x.size(0)
        if batch_n == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
            return

        delta = batch_mean - self.mean
        new_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / new_n)
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * (self.n * batch_n / new_n)
        self.n = new_n

    def finalize(self):
        var = self.M2 / max(self.n - 1, 1)
        std = torch.sqrt(var).clamp_min(1e-8)
        return self.mean.to(torch.float32), std.to(torch.float32)


def normalize_tensor(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    return (x - mu) / sigma


def unnormalize_col(y_norm: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, col: int):
    return y_norm * sigma[col] + mu[col]


# =========================
# Feature engineering
# =========================
def add_temporal_features(df: pd.DataFrame, has_rainfall: bool):
    """
    Expects df has at least: ['node_idx', 'timestep', 'water_level'] and optionally 'rainfall'.
    Adds:
      - cumulative sums: cum_water_level, (cum_rainfall)
      - rolling means over last 12/24/36 timesteps: mean_{k}_water_level, (mean_{k}_rainfall)
    Memory-efficient per-event (groupby node_idx).
    """
    df = df.sort_values(["node_idx", "timestep"]).copy()

    # Water level features
    df["cum_water_level"] = df.groupby("node_idx")["water_level"].cumsum()
    for k in (12, 24, 36):
        df[f"mean_{k}_water_level"] = (
            df.groupby("node_idx")["water_level"]
              .rolling(window=k, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    if has_rainfall:
        df["cum_rainfall"] = df.groupby("node_idx")["rainfall"].cumsum()
        for k in (12, 24, 36):
            df[f"mean_{k}_rainfall"] = (
                df.groupby("node_idx")["rainfall"]
                  .rolling(window=k, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

    return df


# =========================
# Graph builders
# =========================
def idx_builder(timesteps, edgedata, nnodes):
    stack = []
    for t in range(timesteps - 1):
        curedge = edgedata.copy()
        curedge.from_node = curedge.from_node + t * nnodes
        curedge.to_node = curedge.to_node + (t + 1) * nnodes
        stack.append(curedge)
    return torch.tensor(pd.concat(stack).loc[:, ["from_node", "to_node"]].values).int().T


def idx_builder_cross_node(timesteps, edgedata, nnodes1d, nnodes2d):
    stack = []
    for t in range(timesteps - 1):
        curedge = edgedata.copy()
        curedge["from_node"] = curedge.node_2d + t * nnodes2d
        curedge["to_node"] = curedge.node_1d + (t + 1) * nnodes1d
        stack.append(curedge)
    return torch.tensor(pd.concat(stack).loc[:, ["from_node", "to_node"]].values).int().T


def create_directed_temporal_graph(
    start_idx, end_idx,
    nodes1d, nodes2d,
    edges1d, edges2d, edges1d2d,
    edges1dfeats, edges2dfeats,
    node1d_cols, node2d_cols,
    edge1_cols, edge2_cols,
    norm_stats=None,
):
    data = tg.data.HeteroData()

    n1 = nodes1d.loc[(nodes1d.timestep >= start_idx) & (nodes1d.timestep <= end_idx), :]
    n2 = nodes2d.loc[(nodes2d.timestep >= start_idx) & (nodes2d.timestep <= end_idx), :]

    pred_mask_1d = torch.tensor(n1["timestep"].values == n1["timestep"].max(), dtype=torch.bool)
    pred_mask_2d = torch.tensor(n2["timestep"].values == n2["timestep"].max(), dtype=torch.bool)

    x1 = torch.tensor(n1.loc[:, node1d_cols].values, dtype=torch.float32)
    x2 = torch.tensor(n2.loc[:, node2d_cols].values, dtype=torch.float32)

    if norm_stats is not None:
        x1 = normalize_tensor(x1, norm_stats["oneD_mu"], norm_stats["oneD_sigma"])
        x2 = normalize_tensor(x2, norm_stats["twoD_mu"], norm_stats["twoD_sigma"])

    data["oneD"].x = x1
    data["oneD"].num_nodes = x1.size(0)
    data["oneD"].pred_mask = pred_mask_1d

    data["twoD"].x = x2
    data["twoD"].num_nodes = x2.size(0)
    data["twoD"].pred_mask = pred_mask_2d

    nnodes1d = len(nodes1d.node_idx.unique())
    nnodes2d = len(nodes2d.node_idx.unique())
    timesteps = len(range(start_idx, end_idx))

    data["oneD", "oneDedge", "oneD"].edge_index = idx_builder(timesteps, edges1d, nnodes1d)
    e1 = torch.tensor(
        pd.concat([edges1dfeats.loc[:, edge1_cols] for _ in range(timesteps-1)]).values,
        dtype=torch.float32
    )
    if norm_stats is not None:
        e1 = normalize_tensor(e1, norm_stats["edge1_mu"], norm_stats["edge1_sigma"])
    data["oneD", "oneDedge", "oneD"].edge_attr = e1  # (you can rename to .edge_attr later)

    data["twoD", "twoDedge", "twoD"].edge_index = idx_builder(timesteps, edges2d, nnodes2d)
    e2 = torch.tensor(
        pd.concat([edges2dfeats.loc[:, edge2_cols] for _ in range(timesteps-1)]).values,
        dtype=torch.float32
    )
    if norm_stats is not None:
        e2 = normalize_tensor(e2, norm_stats["edge2_mu"], norm_stats["edge2_sigma"])
    data["twoD", "twoDedge", "twoD"].edge_attr = e2

    data["twoD", "twoDoneD", "oneD"].edge_index = idx_builder_cross_node(timesteps, edges1d2d, nnodes1d, nnodes2d)

    data.validate()
    return data


class TemporalGraphStream(IterableDataset):
    def __init__(
        self,
        idxs, nodes1d, nodes2d,
        edges1d, edges2d, edges1d2d,
        edges1dfeats, edges2dfeats,
        node1d_cols, node2d_cols, edge1_cols, edge2_cols,
        norm_stats,
    ):
        super().__init__()
        self.idxs = idxs
        self.nodes1d = nodes1d
        self.nodes2d = nodes2d
        self.edges1d = edges1d
        self.edges2d = edges2d
        self.edges1d2d = edges1d2d
        self.edges1dfeats = edges1dfeats
        self.edges2dfeats = edges2dfeats
        self.node1d_cols = node1d_cols
        self.node2d_cols = node2d_cols
        self.edge1_cols = edge1_cols
        self.edge2_cols = edge2_cols
        self.norm_stats = norm_stats

    def __iter__(self):
        for start_idx, end_idx in self.idxs:
            yield create_directed_temporal_graph(
                start_idx, end_idx,
                self.nodes1d, self.nodes2d,
                self.edges1d, self.edges2d, self.edges1d2d,
                self.edges1dfeats, self.edges2dfeats,
                self.node1d_cols, self.node2d_cols,
                self.edge1_cols, self.edge2_cols,
                norm_stats=self.norm_stats
            )


# =========================
# Load static + normalize static tables (except IDs)
# =========================
base_path = "/Users/Lion/Desktop/UrbanFloodModeling/FloodModel/data/Model_1"
static_1d = pd.read_csv(base_path + "/train/1d_nodes_static.csv")
static_2d = pd.read_csv(base_path + "/train/2d_nodes_static.csv")
edges1dfeats = pd.read_csv(base_path + "/train/1d_edges_static.csv")
edges2dfeats = pd.read_csv(base_path + "/train/2d_edges_static.csv")
edges1d = pd.read_csv(base_path + "/train/1d_edge_index.csv")
edges2d = pd.read_csv(base_path + "/train/2d_edge_index.csv")
edges1d2d = pd.read_csv(base_path + "/train/1d2d_connections.csv")

# Identify ID cols
NODE_ID_COL = "node_idx"
EDGE_ID_COL = "edge_idx" if "edge_idx" in edges1dfeats.columns else edges1dfeats.columns[0]

# Normalize static node features (except node_idx) BEFORE merge
def zscore_df_inplace(df: pd.DataFrame, id_col: str):
    feat_cols = [c for c in df.columns if c != id_col]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    mu = x.mean(dim=0)
    sigma = x.std(dim=0).clamp_min(1e-8)
    df.loc[:, feat_cols] = ((x - mu) / sigma).numpy()
    return feat_cols  # return normalized cols if you want

_ = zscore_df_inplace(static_1d, NODE_ID_COL)
_ = zscore_df_inplace(static_2d, NODE_ID_COL)

# Normalize static edge features (except edge_idx) BEFORE iteration
def zscore_edge_df_inplace(df: pd.DataFrame, id_col: str):
    feat_cols = [c for c in df.columns if c != id_col]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    mu = x.mean(dim=0)
    sigma = x.std(dim=0).clamp_min(1e-8)
    df.loc[:, feat_cols] = ((x - mu) / sigma).numpy()
    return feat_cols

edge1_cols = zscore_edge_df_inplace(edges1dfeats, EDGE_ID_COL)
edge2_cols = zscore_edge_df_inplace(edges2dfeats, EDGE_ID_COL)

event_dirs = sorted(
    glob.glob(base_path + "/train/event_*"),
    key=lambda p: int(Path(p).name.split("_")[-1])
)

# =========================
# PASS 1: streaming stats for node features AFTER:
#   - merge (static already normalized)
#   - feature engineering (running sums + rolling means)
#   - drop node_idx from x columns (keep it for indexing/merge)
# =========================
# Determine feature columns from first event after feature engineering
f0 = event_dirs[0]
n1 = pd.read_csv(f0 + "/1d_nodes_dynamic_all.csv")
n2 = pd.read_csv(f0 + "/2d_nodes_dynamic_all.csv")

# Assumptions about dynamic cols:
#  oneD: has at least node_idx, timestep, water_level
#  twoD: has at least node_idx, timestep, rainfall, water_level
n1 = n1.copy()
n2 = n2.copy()

n1 = pd.merge(n1, static_1d, on=NODE_ID_COL, how="left")
n2 = pd.merge(n2, static_2d, on=NODE_ID_COL, how="left")

n1 = add_temporal_features(n1, has_rainfall=False)
n2 = add_temporal_features(n2, has_rainfall=True)

node1d_cols = [c for c in n1.columns if c != NODE_ID_COL]
node2d_cols = [c for c in n2.columns if c != NODE_ID_COL]

rms_oneD = RunningMeanStd(len(node1d_cols))
rms_twoD = RunningMeanStd(len(node2d_cols))

for f in event_dirs[:10]:
    n1 = pd.read_csv(f + "/1d_nodes_dynamic_all.csv")
    n2 = pd.read_csv(f + "/2d_nodes_dynamic_all.csv")

    n1 = pd.merge(n1, static_1d, on=NODE_ID_COL, how="left")
    n2 = pd.merge(n2, static_2d, on=NODE_ID_COL, how="left")

    n1 = add_temporal_features(n1, has_rainfall=False)
    n2 = add_temporal_features(n2, has_rainfall=True)

    x1 = torch.tensor(n1.loc[:, node1d_cols].values, dtype=torch.float32)
    x2 = torch.tensor(n2.loc[:, node2d_cols].values, dtype=torch.float32)

    rms_oneD.update(x1)
    rms_twoD.update(x2)

oneD_mu, oneD_sigma = rms_oneD.finalize()
twoD_mu, twoD_sigma = rms_twoD.finalize()

# Edge stats (even though edges were z-scored already, we keep these for the graph-level normalize step above)
# If you prefer, you can set mu=0,sigma=1 here since edges are already normalized.
e1 = torch.tensor(edges1dfeats.loc[:, edge1_cols].values, dtype=torch.float32)
e2 = torch.tensor(edges2dfeats.loc[:, edge2_cols].values, dtype=torch.float32)
edge1_mu = e1.mean(dim=0); edge1_sigma = e1.std(dim=0).clamp_min(1e-8)
edge2_mu = e2.mean(dim=0); edge2_sigma = e2.std(dim=0).clamp_min(1e-8)

norm_stats = {
    "oneD_mu": oneD_mu, "oneD_sigma": oneD_sigma,
    "twoD_mu": twoD_mu, "twoD_sigma": twoD_sigma,
    "edge1_mu": edge1_mu, "edge1_sigma": edge1_sigma,
    "edge2_mu": edge2_mu, "edge2_sigma": edge2_sigma,
}

# (Optional) column index for unnormalizing water_level in each node type:
oneD_water_col = node1d_cols.index("water_level") if "water_level" in node1d_cols else None
twoD_water_col = node2d_cols.index("water_level") if "water_level" in node2d_cols else None
twoD_rain_col  = node2d_cols.index("rainfall") if "rainfall" in node2d_cols else None

# =========================
# PASS 2: build datasets (no caching of all events)
# =========================

datasets = []
for f in event_dirs[:10]:
    nodes1d = pd.read_csv(f + "/1d_nodes_dynamic_all.csv")
    nodes2d = pd.read_csv(f + "/2d_nodes_dynamic_all.csv")

    nodes1d = pd.merge(nodes1d, static_1d, on=NODE_ID_COL, how="left")
    nodes2d = pd.merge(nodes2d, static_2d, on=NODE_ID_COL, how="left")

    nodes1d = add_temporal_features(nodes1d, has_rainfall=False)
    nodes2d = add_temporal_features(nodes2d, has_rainfall=True)

    # windows: (i, i+11) means 11-step history if your timestep spacing is 1
    idxs = [(i, i + 11) for i in range(len(nodes1d.timestep.unique()) - 10)]

    datasets.append(
        TemporalGraphStream(
            idxs, nodes1d, nodes2d,
            edges1d, edges2d, edges1d2d,
            edges1dfeats, edges2dfeats,
            node1d_cols, node2d_cols, edge1_cols, edge2_cols,
            norm_stats
        )
    )

dataset_all = ChainDataset(datasets)
DL_NUM_WORKERS = min(2, os.cpu_count() or 0)
DL_PIN_MEMORY = False
DL_PREFETCH = 1
DL_PERSISTENT = DL_NUM_WORKERS > 0

dl_kwargs = {
    "batch_size": 8,
    "shuffle": False,
    "drop_last": True,
    "num_workers": DL_NUM_WORKERS,
    "pin_memory": DL_PIN_MEMORY,
    "persistent_workers": DL_PERSISTENT,
}
if DL_NUM_WORKERS > 0:
    dl_kwargs["prefetch_factor"] = DL_PREFETCH

dl = DataLoader(dataset_all, **dl_kwargs)
