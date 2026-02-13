import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import IterableDataset, ChainDataset
from torch_geometric.loader import DataLoader
import torch_geometric as tg
import glob
import random
from pathlib import Path
from normalization import FeatureNormalizer

# Reduce log noise from per-feature normalization prints
NORMALIZATION_VERBOSE = False

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
        # Clamp negative variance (numerical errors) before sqrt
        var = torch.clamp(var, min=1e-16)
        std = torch.sqrt(var).clamp_min(1e-8)
        
        # Handle NaN/Inf
        mean = self.mean.to(torch.float32)
        std = std.to(torch.float32)
        
        # Replace NaN/Inf with safe defaults
        mean = torch.where(torch.isnan(mean) | torch.isinf(mean), torch.zeros_like(mean), mean)
        std = torch.where(torch.isnan(std) | torch.isinf(std), torch.ones_like(std), std)
        
        return mean, std


def normalize_tensor(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """Normalize tensor (currently identity since pre-normalized to [0,1])."""
    return (x - mu) / sigma


def unnormalize_col(y_norm: torch.Tensor, norm_stats: dict, col: int, node_type: str):
    """
    Unnormalize a specific column using feature-aware params.
    
    Args:
        y_norm: Normalized values [0, 1] or tensor
        norm_stats: Dict containing all normalization params including normalizers
        col: Column index
        node_type: 'oneD' or 'twoD'
    
    Returns:
        Unnormalized values in original scale
    """
    # Get column name and normalizer
    if node_type == 'oneD':
        col_names = norm_stats['node1d_cols']
        normalizer = norm_stats['normalizer_1d']
    elif node_type == 'twoD':
        col_names = norm_stats['node2d_cols']
        normalizer = norm_stats['normalizer_2d']
    else:
        raise ValueError(f"Unknown node_type: {node_type}")
    
    col_name = col_names[col]
    
    # Use the normalizer's unnormalize method
    # Determine if static or dynamic
    if col_name in normalizer.static_features:
        feature_type = 'static'
    else:
        feature_type = 'dynamic'
    
    return normalizer.unnormalize(y_norm, col_name, feature_type)


# =========================
# Feature engineering
# =========================
def add_temporal_features(df: pd.DataFrame, has_rainfall: bool):
    """
    Expects df has at least: ['node_idx', 'timestep', 'water_level'] and optionally 'rainfall'.
    Adds features built strictly from history by shifting by 1 timestep:
      - cumulative sums: cum_water_level, (cum_rainfall)
      - rolling means over last 12/24/36 timesteps: mean_{k}_water_level, (mean_{k}_rainfall)
    Memory-efficient per-event (groupby node_idx).
    """
    # Use raw timestep for ordering if available
    tcol = "timestep_raw" if "timestep_raw" in df.columns else "timestep"
    df = df.sort_values(["node_idx", tcol]).copy()

    # Shift by 1 to avoid using current timestep in temporal features
    wl_hist = df.groupby("node_idx")["water_level"].shift(1)
    wl_hist = wl_hist.fillna(0.0)

    # Water level features (log-transform cumulative to prevent unbounded growth)
    cum_wl = wl_hist.groupby(df["node_idx"]).cumsum().values
    df["cum_water_level"] = np.sign(cum_wl) * np.log1p(np.abs(cum_wl))

    for k in (12, 24, 36):
        df[f"mean_{k}_water_level"] = (
            wl_hist.groupby(df["node_idx"])
                  .rolling(window=k, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
        )

    if has_rainfall and "rainfall" in df.columns:
        rf_hist = df.groupby("node_idx")["rainfall"].shift(1)
        rf_hist = rf_hist.fillna(0.0)

        cum_rf = rf_hist.groupby(df["node_idx"]).cumsum().values
        df["cum_rainfall"] = np.sign(cum_rf) * np.log1p(np.abs(cum_rf))

        for k in (12, 24, 36):
            df[f"mean_{k}_rainfall"] = (
                rf_hist.groupby(df["node_idx"])
                      .rolling(window=k, min_periods=1)
                      .mean()
                      .reset_index(level=0, drop=True)
            )

    return df


def preprocess_2d_nodes(nodes2d_df):
    """
    Preprocess 2D nodes:
    - Fill missing min_elevation with elevation values
    - Drop aspect and curvature columns
    - Use KNN to interpolate area for zero/near-zero values
    """
    from sklearn.neighbors import NearestNeighbors
    
    df = nodes2d_df.copy()
    
    # Fill min_elevation with elevation if missing
    if "min_elevation" in df.columns and "elevation" in df.columns:
        mask = pd.isna(df["min_elevation"])
        df.loc[mask, "min_elevation"] = df.loc[mask, "elevation"]
    
    # Drop aspect and curvature if they exist
    drop_cols = [c for c in ["aspect", "curvature"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # KNN interpolation for zero/near-zero area
    if "area" in df.columns and ("position_x" in df.columns or "x" in df.columns):
        pos_x_col = "position_x" if "position_x" in df.columns else "x"
        pos_y_col = "position_y" if "position_y" in df.columns else "y"
        
        # Identify near-zero area rows (area < 1e-6)
        zero_mask = df["area"].abs() < 1e-6
        if zero_mask.any():
            # Get positions for all nodes
            positions = df[[pos_x_col, pos_y_col]].values
            
            # Get non-zero area nodes for interpolation
            nonzero_idx = (~zero_mask).values
            if nonzero_idx.sum() > 0:
                nonzero_positions = positions[nonzero_idx]
                nonzero_areas = df.loc[~zero_mask, "area"].values
                
                # KNN: find k nearest neighbors (cap at available)
                k = min(5, nonzero_idx.sum())
                if k > 0:
                    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
                    knn.fit(nonzero_positions)
                    
                    # Find zero-area node positions
                    zero_positions = positions[zero_mask]
                    distances, indices = knn.kneighbors(zero_positions)
                    
                    # Average area of k nearest neighbors
                    interpolated_areas = nonzero_areas[indices].mean(axis=1)
                    
                    # Fill NaN/zero areas with interpolated values
                    zero_rows = df.index[zero_mask].tolist()
                    for i, row_idx in enumerate(zero_rows):
                        df.loc[row_idx, "area"] = interpolated_areas[i]
    
    return df


# =========================
# Graph builders
# =========================
def idx_builder(timesteps, edgedata, nnodes):
    """
    Build temporal edges for time-unrolled graph.
    Assumes edgedata has 'from_node' and 'to_node' columns with 0-indexed node IDs.
    Maps edges across timesteps: (node_i at t) -> (node_j at t+1)
    
    Args:
        timesteps: number of timesteps in window
        edgedata: DataFrame with from_node, to_node (node IDs within a single timestep)
        nnodes: number of nodes per timestep
    
    Returns:
        edge_index [2, E] where E = len(edgedata) * (timesteps - 1)
    """
    stack = []
    for t in range(timesteps - 1):
        curedge = edgedata.copy()
        # Map from timestep t to timestep t+1
        # Row index = timestep_offset + node_id
        curedge['from_node'] = curedge['from_node'] + t * nnodes
        curedge['to_node'] = curedge['to_node'] + (t + 1) * nnodes
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

    # Use raw timestep for slicing if available (normalized timestep is in [0,1])
    tcol1 = "timestep_raw" if "timestep_raw" in nodes1d.columns else "timestep"
    tcol2 = "timestep_raw" if "timestep_raw" in nodes2d.columns else "timestep"

    n1 = nodes1d.loc[(nodes1d[tcol1] >= start_idx) & (nodes1d[tcol1] <= end_idx), :]
    n2 = nodes2d.loc[(nodes2d[tcol2] >= start_idx) & (nodes2d[tcol2] <= end_idx), :]

    if n1.empty or n2.empty:
        return None

    # Sort by timestep, then node_idx to ensure consistent row ordering
    n1 = n1.sort_values([tcol1, 'node_idx']).reset_index(drop=True)
    n2 = n2.sort_values([tcol2, 'node_idx']).reset_index(drop=True)

    pred_mask_1d = torch.tensor(n1[tcol1].values == n1[tcol1].max(), dtype=torch.bool)
    pred_mask_2d = torch.tensor(n2[tcol2].values == n2[tcol2].max(), dtype=torch.bool)

    x1 = torch.tensor(n1.loc[:, node1d_cols].values, dtype=torch.float32)
    x2 = torch.tensor(n2.loc[:, node2d_cols].values, dtype=torch.float32)

    if norm_stats is not None:
        x1 = normalize_tensor(x1, norm_stats["oneD_mu"], norm_stats["oneD_sigma"])
        x2 = normalize_tensor(x2, norm_stats["twoD_mu"], norm_stats["twoD_sigma"])

    # CRITICAL: Store ground truth labels BEFORE masking inputs to prevent data leakage
    water_col_1d = node1d_cols.index('water_level') if 'water_level' in node1d_cols else None
    water_col_2d = node2d_cols.index('water_level') if 'water_level' in node2d_cols else None
    
    # Calculate nodes per timestep (needed for masking)
    nnodes1d = len(n1.node_idx.unique())
    nnodes2d = len(n2.node_idx.unique())
    
    # Extract ground truth water_level for prediction nodes (normalized)
    y1 = x1[pred_mask_1d][:, water_col_1d].clone() if water_col_1d is not None else None
    y2 = x2[pred_mask_2d][:, water_col_2d].clone() if water_col_2d is not None else None
    
    # Safety check: ensure labels are finite
    if y1 is not None and not torch.isfinite(y1).all():
        print(f"[WARN] Non-finite labels in 1D nodes for window [{start_idx}, {end_idx}]")
        return None
    if y2 is not None and not torch.isfinite(y2).all():
        print(f"[WARN] Non-finite labels in 2D nodes for window [{start_idx}, {end_idx}]")
        return None
    
    # Now mask out future information from inputs (VECTORIZED for speed)
    # For nodes at final timestep (pred_mask=True), replace current water_level with previous timestep's value
    if water_col_1d is not None:
        # Find indices of prediction nodes
        pred_indices = pred_mask_1d.nonzero(as_tuple=True)[0]
        # Calculate previous timestep indices (shift back by nnodes1d)
        prev_indices = pred_indices - nnodes1d
        # Only copy where previous timestep exists (prev_indices >= 0)
        valid_mask = prev_indices >= 0
        if valid_mask.any():
            x1[pred_indices[valid_mask], water_col_1d] = x1[prev_indices[valid_mask], water_col_1d]
        # Set to 0 where no previous timestep (shouldn't happen with 11 timestep windows)
        if (~valid_mask).any():
            x1[pred_indices[~valid_mask], water_col_1d] = 0.0
    
    if water_col_2d is not None:
        # Find indices of prediction nodes
        pred_indices = pred_mask_2d.nonzero(as_tuple=True)[0]
        # Calculate previous timestep indices (shift back by nnodes2d)
        prev_indices = pred_indices - nnodes2d
        # Only copy where previous timestep exists (prev_indices >= 0)
        valid_mask = prev_indices >= 0
        if valid_mask.any():
            x2[pred_indices[valid_mask], water_col_2d] = x2[prev_indices[valid_mask], water_col_2d]
        # Set to 0 where no previous timestep (shouldn't happen with 11 timestep windows)
        if (~valid_mask).any():
            x2[pred_indices[~valid_mask], water_col_2d] = 0.0
    
    data["oneD"].x = x1
    data["oneD"].y = y1  # Ground truth labels (normalized, only for pred nodes)
    data["oneD"].num_nodes = x1.size(0)
    data["oneD"].pred_mask = pred_mask_1d
    # Store base_area for flux computation (aligned with node rows)
    base_areas = torch.tensor(n1["base_area"].values, dtype=torch.float32)
    data["oneD"].base_area = base_areas  # [num_nodes_in_batch]

    data["twoD"].x = x2
    data["twoD"].y = y2  # Ground truth labels (normalized, only for pred nodes)
    data["twoD"].num_nodes = x2.size(0)
    data["twoD"].pred_mask = pred_mask_2d
    # Store cell_area for flux computation (aligned with node rows)
    cell_areas = torch.tensor(n2["area"].values, dtype=torch.float32)
    data["twoD"].cell_area = cell_areas  # [num_nodes_in_batch]

    # Use unique nodes in BATCH, not entire dataset (already calculated above)
    timesteps = end_idx - start_idx + 1

    if nnodes1d == 0 or nnodes2d == 0:
        return None
    
    # Check if data is complete (all nodes present in all timesteps)
    expected_1d_rows = nnodes1d * timesteps
    expected_2d_rows = nnodes2d * timesteps
    
    # If data incomplete, skip this window
    if len(n1) != expected_1d_rows or len(n2) != expected_2d_rows:
        print(f"Skipping incomplete window [{start_idx}, {end_idx}]: 1D {len(n1)}/{expected_1d_rows}, 2D {len(n2)}/{expected_2d_rows}")
        return None  # Signal to skip this window

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

    # Store num_timesteps on node types so it survives batching
    data["oneD"].num_timesteps = torch.tensor(timesteps, dtype=torch.long)
    data["twoD"].num_timesteps = torch.tensor(timesteps, dtype=torch.long)

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
        shuffle=False,
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
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(self.idxs)
        if self.shuffle:
            random.shuffle(idxs)
        for idx_pair in idxs:
            start_idx, end_idx = idx_pair
            data = create_directed_temporal_graph(
                start_idx, end_idx,
                self.nodes1d, self.nodes2d,
                self.edges1d, self.edges2d, self.edges1d2d,
                self.edges1dfeats, self.edges2dfeats,
                self.node1d_cols, self.node2d_cols,
                self.edge1_cols, self.edge2_cols,
                norm_stats=self.norm_stats
            )
            
            # Skip incomplete windows
            if data is None:
                continue

            yield data


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

# Normalize static edge features using FeatureNormalizer
normalizer_edge1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
normalizer_edge2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)

normalizer_edge1d.fit_static(edges1dfeats.copy(), EDGE_ID_COL, skew_threshold=2.0)
normalizer_edge2d.fit_static(edges2dfeats.copy(), EDGE_ID_COL, skew_threshold=2.0)

edges1dfeats = normalizer_edge1d.transform_static(edges1dfeats, EDGE_ID_COL)
edges2dfeats = normalizer_edge2d.transform_static(edges2dfeats, EDGE_ID_COL)

edge1_cols = [c for c in edges1dfeats.columns if c != EDGE_ID_COL]
edge2_cols = [c for c in edges2dfeats.columns if c != EDGE_ID_COL]

# Store edge column info globally for model access
edge_col_info = {
    'edge1_cols': edge1_cols,
    'edge2_cols': edge2_cols,
}

event_dirs = sorted(
    glob.glob(base_path + "/train/event_*"),
    key=lambda p: int(Path(p).name.split("_")[-1])
)

# =========================
# PASS 1: Feature-aware normalization (static vs dynamic, separate domains)
# Dynamic features to exclude from training: water_volume (2D), inlet_flow (1D)
# =========================

print("[INFO] ===== Feature Normalization Setup =====")

# Define exclusions
EXCLUDE_1D_DYNAMIC = ['inlet_flow']  # Exclude from 1D dynamic
EXCLUDE_2D_DYNAMIC = ['water_volume']  # Exclude from 2D dynamic

# Initialize separate normalizers for 1D and 2D
normalizer_1d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
normalizer_2d = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)

# Preprocess static 2D nodes BEFORE normalization (fill NaN, drop cols, interpolate area)
print("[INFO] Preprocessing static 2D nodes...")
static_2d = preprocess_2d_nodes(static_2d)

# Fit and transform static features (small, already in memory)
print("[INFO] Fitting static feature normalization...")
normalizer_1d.fit_static(static_1d.copy(), NODE_ID_COL, skew_threshold=2.0)
normalizer_2d.fit_static(static_2d.copy(), NODE_ID_COL, skew_threshold=2.0)

static_1d = normalizer_1d.transform_static(static_1d, NODE_ID_COL)
static_2d = normalizer_2d.transform_static(static_2d, NODE_ID_COL)

# Stream through events to fit dynamic normalization (memory efficient)
print(f"[INFO] Streaming through {len(event_dirs)} events for dynamic normalization...")

# Get dynamic feature columns from first event
n1_temp = pd.read_csv(event_dirs[0] + "/1d_nodes_dynamic_all.csv")
n2_temp = pd.read_csv(event_dirs[0] + "/2d_nodes_dynamic_all.csv")

# Drop excluded columns
n1_temp = n1_temp.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_temp.columns])
n2_temp = n2_temp.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_temp.columns])

base_1d_dynamic_feats = [c for c in n1_temp.columns if c not in [NODE_ID_COL, 'timestep']]
base_2d_dynamic_feats = [c for c in n2_temp.columns if c not in [NODE_ID_COL, 'timestep', 'rainfall']]

# Add rainfall for normalization (do not normalize timestep)
dynamic_1d_cols = base_1d_dynamic_feats
dynamic_2d_cols = base_2d_dynamic_feats + ['rainfall']

print(f"[INFO] 1D static features: {normalizer_1d.static_features}")
print(f"[INFO] 2D static features: {normalizer_2d.static_features}")
print(f"[INFO] 1D base dynamic features: {base_1d_dynamic_feats}")
print(f"[INFO] 2D base dynamic features: {base_2d_dynamic_feats}")

# Initialize streaming
print("[INFO] Fitting dynamic feature normalization (streaming)...")
normalizer_1d.init_dynamic_streaming(dynamic_1d_cols, exclude_cols=None)
normalizer_2d.init_dynamic_streaming(dynamic_2d_cols, exclude_cols=None)


# Stream through events
for event_dir in event_dirs:
    n1_dyn = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
    n2_dyn = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
    
    # Drop excluded columns
    n1_dyn = n1_dyn.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_dyn.columns])
    n2_dyn = n2_dyn.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_dyn.columns])
    
    # Update streaming stats
    normalizer_1d.update_dynamic_streaming(n1_dyn, exclude_cols=None)
    normalizer_2d.update_dynamic_streaming(n2_dyn, exclude_cols=None)

# Finalize normalization parameters
normalizer_1d.finalize_dynamic_streaming(skew_threshold=2.0)
normalizer_2d.finalize_dynamic_streaming(skew_threshold=2.0)

# Now we need to determine final column structure after temporal feature engineering
# Do a dry run with first event to get engineered feature names
f0 = event_dirs[0]
n1_test = pd.read_csv(f0 + "/1d_nodes_dynamic_all.csv")
n2_test = pd.read_csv(f0 + "/2d_nodes_dynamic_all.csv")

# Drop excluded columns
n1_test = n1_test.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1_test.columns])
n2_test = n2_test.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2_test.columns])

# Apply dynamic normalization
n1_test = normalizer_1d.transform_dynamic(n1_test, exclude_cols=None)
n2_test = normalizer_2d.transform_dynamic(n2_test, exclude_cols=None)

# Merge with normalized static features (no re-normalization)
n1_test = pd.merge(n1_test, static_1d, on=NODE_ID_COL, how="left")
n2_test = pd.merge(n2_test, static_2d, on=NODE_ID_COL, how="left")
n2_test = preprocess_2d_nodes(n2_test)

# Add temporal features (these will need normalization too)
n1_test = add_temporal_features(n1_test, has_rainfall=False)
n2_test = add_temporal_features(n2_test, has_rainfall=True)

# Get engineered feature names
engineered_1d = [c for c in n1_test.columns if c.startswith('cum_') or c.startswith('mean_')]
engineered_2d = [c for c in n2_test.columns if c.startswith('cum_') or c.startswith('mean_')]

print(f"[INFO] Engineered 1D features: {engineered_1d}")
print(f"[INFO] Engineered 2D features: {engineered_2d}")

# Fit normalization for engineered features (streaming to avoid memory issues)
print("[INFO] Fitting engineered feature normalization (streaming)...")

# Initialize streaming for engineered features
normalizer_1d_eng = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
normalizer_2d_eng = FeatureNormalizer(verbose=NORMALIZATION_VERBOSE)
normalizer_1d_eng.init_dynamic_streaming(engineered_1d, exclude_cols=None)
normalizer_2d_eng.init_dynamic_streaming(engineered_2d, exclude_cols=None)

# Stream through events
for event_dir in event_dirs:
    n1 = pd.read_csv(event_dir + "/1d_nodes_dynamic_all.csv")
    n2 = pd.read_csv(event_dir + "/2d_nodes_dynamic_all.csv")
    
    # Drop excluded
    n1 = n1.drop(columns=[c for c in EXCLUDE_1D_DYNAMIC if c in n1.columns])
    n2 = n2.drop(columns=[c for c in EXCLUDE_2D_DYNAMIC if c in n2.columns])
    
    # Apply dynamic norm
    n1 = normalizer_1d.transform_dynamic(n1, exclude_cols=None)
    n2 = normalizer_2d.transform_dynamic(n2, exclude_cols=None)
    
    # Merge with static (already normalized)
    n1 = pd.merge(n1, static_1d, on=NODE_ID_COL, how="left")
    n2 = pd.merge(n2, static_2d, on=NODE_ID_COL, how="left")
    n2 = preprocess_2d_nodes(n2)
    
    # Add temporal features
    n1 = add_temporal_features(n1, has_rainfall=False)
    n2 = add_temporal_features(n2, has_rainfall=True)
    
    # Update streaming stats for engineered features
    normalizer_1d_eng.update_dynamic_streaming(n1[engineered_1d], exclude_cols=None)
    normalizer_2d_eng.update_dynamic_streaming(n2[engineered_2d], exclude_cols=None)

# Finalize engineered feature normalization
normalizer_1d_eng.finalize_dynamic_streaming(skew_threshold=2.0)
normalizer_2d_eng.finalize_dynamic_streaming(skew_threshold=2.0)

# Merge engineered params into main normalizer
normalizer_1d.dynamic_params.update(normalizer_1d_eng.dynamic_params)
normalizer_2d.dynamic_params.update(normalizer_2d_eng.dynamic_params)
normalizer_1d.dynamic_features.extend(engineered_1d)
normalizer_2d.dynamic_features.extend(engineered_2d)

# Final column order (after all transformations)
# Exclude ids and timestep fields from model features
exclude_node_cols = {NODE_ID_COL, 'timestep', 'timestep_raw'}
node1d_cols = [c for c in n1_test.columns if c not in exclude_node_cols]
node2d_cols = [c for c in n2_test.columns if c not in exclude_node_cols]

# Create feature type mappings for unnormalization
feature_type_1d = {}
feature_type_2d = {}

for col in node1d_cols:
    if col in normalizer_1d.static_features:
        feature_type_1d[col] = 'static'
    else:
        feature_type_1d[col] = 'dynamic'

for col in node2d_cols:
    if col in normalizer_2d.static_features:
        feature_type_2d[col] = 'static'
    else:
        feature_type_2d[col] = 'dynamic'

print(f"[INFO] Final 1D features: {len(node1d_cols)} ({sum(1 for v in feature_type_1d.values() if v=='static')} static, {sum(1 for v in feature_type_1d.values() if v=='dynamic')} dynamic)")
print(f"[INFO] Final 2D features: {len(node2d_cols)} ({sum(1 for v in feature_type_2d.values() if v=='static')} static, {sum(1 for v in feature_type_2d.values() if v=='dynamic')} dynamic)")

# Create mu/sigma tensors for normalize_tensor compatibility (all zeros/ones since already normalized)
oneD_mu = torch.zeros(len(node1d_cols))
oneD_sigma = torch.ones(len(node1d_cols))
twoD_mu = torch.zeros(len(node2d_cols))
twoD_sigma = torch.ones(len(node2d_cols))

# Edge normalization (keep existing approach)
edge1_mu = torch.zeros(len(edge1_cols))
edge1_sigma = torch.ones(len(edge1_cols))
edge2_mu = torch.zeros(len(edge2_cols))
edge2_sigma = torch.ones(len(edge2_cols))

norm_stats = {
    "oneD_mu": oneD_mu, "oneD_sigma": oneD_sigma,
    "twoD_mu": twoD_mu, "twoD_sigma": twoD_sigma,
    "edge1_mu": edge1_mu, "edge1_sigma": edge1_sigma,
    "edge2_mu": edge2_mu, "edge2_sigma": edge2_sigma,
    # Feature-aware normalization params for unnormalization
    "static_1d_params": normalizer_1d.static_params,
    "static_2d_params": normalizer_2d.static_params,
    "dynamic_1d_params": normalizer_1d.dynamic_params,
    "dynamic_2d_params": normalizer_2d.dynamic_params,
    "feature_type_1d": feature_type_1d,
    "feature_type_2d": feature_type_2d,
    "node1d_cols": node1d_cols,
    "node2d_cols": node2d_cols,
    "exclude_1d": EXCLUDE_1D_DYNAMIC,
    "exclude_2d": EXCLUDE_2D_DYNAMIC,
    "normalizer_1d": normalizer_1d,
    "normalizer_2d": normalizer_2d,
}

# (Optional) column index for unnormalizing water_level in each node type:
oneD_water_col = node1d_cols.index("water_level") if "water_level" in node1d_cols else None
twoD_water_col = node2d_cols.index("water_level") if "water_level" in node2d_cols else None
twoD_rain_col  = node2d_cols.index("rainfall") if "rainfall" in node2d_cols else None

# =========================
# PASS 2: build datasets
# =========================

# Create single dataset with all events' windows that can shuffle across events
class MultiEventGraphStream(IterableDataset):
    def __init__(self, event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
                 node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
                 shuffle=True):
        super().__init__()
        self.event_file_list = event_file_list  # List of (event_idx, file_path) tuples
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
        self.static_1d = static_1d
        self.static_2d = static_2d
        self.shuffle = shuffle
        
        # Build flat list of (event_idx, window_pair, file_path) tuples WITHOUT loading data
        self.all_windows = []
        for event_idx, f in event_file_list:
            # Read just to get timestep count, then discard
            nodes1d_temp = pd.read_csv(f + "/1d_nodes_dynamic_all.csv")
            num_timesteps = len(nodes1d_temp.timestep.unique())
            del nodes1d_temp  # Free memory immediately
            
            # Generate window indices (11 timesteps: from i to i+10 inclusive)
            idxs = [(i, i + 10) for i in range(num_timesteps - 10)]
            for window_pair in idxs:
                self.all_windows.append((event_idx, window_pair, f))
    
    def __iter__(self):
        windows = list(self.all_windows)
        if self.shuffle:
            random.shuffle(windows)  # Shuffle across all events and windows
        
        for event_idx, window_pair, file_path in windows:
            # Load event data on-demand
            nodes1d = pd.read_csv(file_path + "/1d_nodes_dynamic_all.csv")
            nodes2d = pd.read_csv(file_path + "/2d_nodes_dynamic_all.csv")

            # Preserve raw timestep for window slicing before normalization
            if "timestep" in nodes1d.columns:
                nodes1d["timestep_raw"] = nodes1d["timestep"]
            if "timestep" in nodes2d.columns:
                nodes2d["timestep_raw"] = nodes2d["timestep"]
            
            # Drop excluded columns
            exclude_1d = self.norm_stats.get('exclude_1d', [])
            exclude_2d = self.norm_stats.get('exclude_2d', [])
            nodes1d = nodes1d.drop(columns=[c for c in exclude_1d if c in nodes1d.columns])
            nodes2d = nodes2d.drop(columns=[c for c in exclude_2d if c in nodes2d.columns])
            
            # Apply dynamic normalization (before merge with static)
            normalizer_1d = self.norm_stats['normalizer_1d']
            normalizer_2d = self.norm_stats['normalizer_2d']
            nodes1d = normalizer_1d.transform_dynamic(nodes1d, exclude_cols=None)
            nodes2d = normalizer_2d.transform_dynamic(nodes2d, exclude_cols=None)
            
            # Merge with pre-normalized static features
            nodes1d = pd.merge(nodes1d, self.static_1d, on="node_idx", how="left")
            nodes2d = pd.merge(nodes2d, self.static_2d, on="node_idx", how="left")
            nodes2d = preprocess_2d_nodes(nodes2d)

            # Add temporal features
            nodes1d = add_temporal_features(nodes1d, has_rainfall=False)
            nodes2d = add_temporal_features(nodes2d, has_rainfall=True)
            
            # Normalize engineered temporal features
            engineered_1d = [c for c in nodes1d.columns if c.startswith('cum_') or c.startswith('mean_')]
            engineered_2d = [c for c in nodes2d.columns if c.startswith('cum_') or c.startswith('mean_')]
            
            for col in engineered_1d:
                if col in normalizer_1d.dynamic_params:
                    vals = nodes1d[col].astype(float).values
                    params = normalizer_1d.dynamic_params[col]
                    
                    if params['log']:
                        vals = np.log1p(np.abs(vals)) * np.sign(vals)
                    
                    if params['max'] > params['min']:
                        vals = (vals - params['min']) / (params['max'] - params['min'])
                    else:
                        vals = np.zeros_like(vals)
                    
                    # Convert column to float to avoid dtype errors
                    nodes1d[col] = vals.astype(np.float32)
            
            for col in engineered_2d:
                if col in normalizer_2d.dynamic_params:
                    vals = nodes2d[col].astype(float).values
                    params = normalizer_2d.dynamic_params[col]
                    
                    if params['log']:
                        vals = np.log1p(np.abs(vals)) * np.sign(vals)
                    
                    if params['max'] > params['min']:
                        vals = (vals - params['min']) / (params['max'] - params['min'])
                    else:
                        vals = np.zeros_like(vals)
                    
                    # Convert column to float to avoid dtype errors
                    nodes2d[col] = vals.astype(np.float32)
           
            start_idx, end_idx = window_pair
            data = create_directed_temporal_graph(
                start_idx, end_idx,
                nodes1d, nodes2d,
                self.edges1d, self.edges2d, self.edges1d2d,
                self.edges1dfeats, self.edges2dfeats,
                self.node1d_cols, self.node2d_cols,
                self.edge1_cols, self.edge2_cols,
                norm_stats=self.norm_stats
            )
            
            # Skip incomplete windows
            if data is None:
                continue
            
            yield data

# Build event file list without loading data
event_file_list = [(event_idx, f) for event_idx, f in enumerate(event_dirs)]

dataset_all = MultiEventGraphStream(
    event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
    node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
    shuffle=True
)
def _running_in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


DL_NUM_WORKERS = int(
    os.environ.get("FLOOD_FLUENT_WORKERS", "0" if _running_in_notebook() else "0")
)
DL_PIN_MEMORY = False
DL_PREFETCH = 2
DL_PERSISTENT = DL_NUM_WORKERS > 0

dl_kwargs = {
    "batch_size": 8,
    # shuffle is handled inside TemporalGraphStream for IterableDataset
    "drop_last": True,
    "num_workers": DL_NUM_WORKERS,
    "pin_memory": DL_PIN_MEMORY,
    "persistent_workers": DL_PERSISTENT,
}
if DL_NUM_WORKERS > 0:
    dl_kwargs["prefetch_factor"] = DL_PREFETCH


def get_dataloader():
    dataset = MultiEventGraphStream(
        event_file_list, edges1d, edges2d, edges1d2d, edges1dfeats, edges2dfeats,
        node1d_cols, node2d_cols, edge1_cols, edge2_cols, norm_stats, static_1d, static_2d,
        shuffle=True
    )
    return DataLoader(dataset, **dl_kwargs)


dl = DataLoader(dataset_all, **dl_kwargs)
