import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


# ---------------------------------------------------------
# Mass Conservation Update (physics-grounded state update)
# ---------------------------------------------------------

class MassConservationUpdate(nn.Module):
    """
    Mass-conservation style state update on a time-unrolled graph with edges t -> t+1.

    Assumptions (match your construction):
      - Node indices are ordered by time blocks.
      - For a given node type, nodes_per_timestep = num_nodes_total // T is constant.
      - For nodes at timestep t+1, the previous state node index is prev = idx - nodes_per_timestep.

    The update uses:
      - inflow at dst: sum(flux_e) over edges ending at dst
      - outflow at prev: sum(flux_e) over edges starting at prev
      - next_state = prev_state + dt/area * (inflow(dst) - outflow(prev)) + dt * rainfall(prev)

    flux_e is learned but constrained non-negative.
    """

    def __init__(
        self,
        wl_idx: int,
        edge_slope_idx: int | None,
        edge_length_idx: int | None,
        rain_idx: int | None = None,
        dt: float = 1.0,
        use_learned_flux: bool = True,
        hidden_flux: int = 32,
        clamp_nonneg_flux: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.wl_idx = wl_idx
        self.rain_idx = rain_idx
        self.edge_slope_idx = edge_slope_idx
        self.edge_length_idx = edge_length_idx
        self.dt = float(dt)
        self.use_learned_flux = bool(use_learned_flux)
        self.clamp_nonneg_flux = bool(clamp_nonneg_flux)
        self.eps = float(eps)

        # Flux model f([wl_src, wl_prev_dst, slope, inv_length]) -> flux >= 0
        in_dim = 2  # wl_src, wl_prev_dst
        if edge_slope_idx is not None:
            in_dim += 1
        if edge_length_idx is not None:
            in_dim += 1  # we'll feed inv_length

        if self.use_learned_flux:
            self.flux_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_flux),
                nn.ReLU(),
                nn.Linear(hidden_flux, 1),
            )
        else:
            self.flux_mlp = None

    def _compute_flux(self, wl_src, wl_prev_dst, edge_attr):
        """Compute non-negative flux from source to destination."""
        # wl_src, wl_prev_dst: [E, 1]
        feats = [wl_src, wl_prev_dst]

        if self.edge_slope_idx is not None:
            slope = edge_attr[:, self.edge_slope_idx:self.edge_slope_idx+1]
            feats.append(slope)

        if self.edge_length_idx is not None:
            length = edge_attr[:, self.edge_length_idx:self.edge_length_idx+1].clamp_min(self.eps)
            inv_len = 1.0 / length
            feats.append(inv_len)

        z = torch.cat(feats, dim=1)  # [E, in_dim]

        if self.flux_mlp is not None:
            q = self.flux_mlp(z)  # [E, 1]
        else:
            # Simple physically-motivated baseline
            # q ~ relu(wl_src - wl_prev_dst) * relu(slope) * inv_length
            q = F.relu(wl_src - wl_prev_dst)
            if self.edge_slope_idx is not None:
                q = q * F.relu(edge_attr[:, self.edge_slope_idx:self.edge_slope_idx+1])
            if self.edge_length_idx is not None:
                q = q * (1.0 / edge_attr[:, self.edge_length_idx:self.edge_length_idx+1].clamp_min(self.eps))

        if self.clamp_nonneg_flux:
            q = F.relu(q)

        return q  # [E, 1]

    def forward(
        self,
        x: torch.Tensor,              # [N, F] for one node type (time-unrolled nodes)
        edge_index: torch.Tensor,     # [2, E] edges t -> t+1
        edge_attr: torch.Tensor,      # [E, D]
        num_timesteps: int,           # T for this node type in this graph window
        node_area: torch.Tensor | None = None,  # [N] or [N,1], aligned with x rows
    ):
        src = edge_index[0]
        dst = edge_index[1]

        # nodes per timestep for this node type
        N = x.size(0)
        T = int(num_timesteps)
        n_per_t = N // T  # assumes constant per time slice

        # prev index for each dst (dst is at t+1)
        prev = dst.to(torch.long) - n_per_t

        # Identify valid dst (i.e., not the first time block)
        valid = prev >= 0

        # Pull water levels for flux computation
        wl_src = x[src, self.wl_idx].unsqueeze(1)  # [E,1]

        # For wl_prev_dst, only valid edges make sense; fill invalid with dst wl
        wl_prev_dst = torch.zeros_like(wl_src)
        wl_prev_dst[valid] = x[prev[valid], self.wl_idx].unsqueeze(1)
        wl_prev_dst[~valid] = x[dst[~valid], self.wl_idx].unsqueeze(1)

        # Compute per-edge flux q_e >= 0
        q = self._compute_flux(wl_src, wl_prev_dst, edge_attr)  # [E,1]

        # Aggregate inflow to dst
        inflow_dst = pyg_utils.scatter(q, dst, dim=0, dim_size=N, reduce="sum")  # [N,1]

        # Aggregate outflow from src
        outflow_src = pyg_utils.scatter(q, src, dim=0, dim_size=N, reduce="sum")  # [N,1]

        # Build prev_state for every node
        prev_state = x[:, self.wl_idx].clone().unsqueeze(1)  # [N,1]
        idxs = torch.arange(N, device=x.device)
        has_prev = idxs >= n_per_t
        prev_state[has_prev] = x[idxs[has_prev] - n_per_t, self.wl_idx].unsqueeze(1)

        # Area scaling
        if node_area is None:
            area = torch.ones((N, 1), device=x.device, dtype=x.dtype)
        else:
            area = node_area
            if area.dim() == 1:
                area = area.unsqueeze(1)
            # More aggressive clamping for numerical stability
            area = area.to(x.dtype).clamp_min(0.1)  # Clamp to minimum 0.1 instead of eps

        # Safety: check for invalid areas
        if torch.any(torch.isnan(area)):
            print(f"WARNING: NaN in node_area, using ones")
            area = torch.ones_like(area)
        if torch.any(torch.isinf(area)):
            print(f"WARNING: Inf in node_area, using ones")
            area = torch.ones_like(area)

        # Outflow for each node should be taken from its prev node
        outflow_prev = torch.zeros_like(inflow_dst)
        outflow_prev[has_prev] = outflow_src[idxs[has_prev] - n_per_t]

        net = inflow_dst - outflow_prev  # [N,1]

        # Rainfall source term
        rain_term = 0.0
        if self.rain_idx is not None:
            rain = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
            rain[has_prev] = x[idxs[has_prev] - n_per_t, self.rain_idx].unsqueeze(1)
            rain_term = self.dt * rain

        wl_next = prev_state + (self.dt / area) * net + rain_term  # [N,1]
        
        # Guard against NaN/Inf in output
        if torch.any(torch.isnan(wl_next)):
            print(f"WARNING: NaN detected in wl_next, replacing with prev_state")
            wl_next = torch.where(torch.isnan(wl_next), prev_state, wl_next)
        if torch.any(torch.isinf(wl_next)):
            print(f"WARNING: Inf detected in wl_next, clamping")
            wl_next = torch.clamp(wl_next, -1e6, 1e6)

        return wl_next, q




# ---------------------------------------------------------
# Homogeneous backbone (edge-feature aware)
# ---------------------------------------------------------

class EdgeAwareTransformerBackbone(nn.Module):
    def __init__(self, hidden=128, heads=2, num_layers=3, dropout=0.3, edge_dim=1):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                pyg_nn.TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden,
                    heads=heads,
                    concat=False,  # Average heads instead of concat for stable gradients
                    dropout=dropout,
                    edge_dim=edge_dim,
                    bias=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.input_norm(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = F.elu(h)
            h = self.drop(h)
            x = norm(x + h)
        return x


# ---------------------------------------------------------
# Hetero wrapper
# ---------------------------------------------------------

class HeteroEdgeAwareTransformer(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims_by_type,
        edge_dim_by_type,
        out_dim,
        hidden=128,
        heads=2,
        num_layers=3,
        dropout=0.3,
        aggr="sum",
        use_cons_of_mass=False,
        flux_indices=None,
        norm_stats=None,
    ):
        super().__init__()

        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.use_cons_of_mass = use_cons_of_mass
        self.norm_stats = norm_stats

        # ---- node input projections ----
        self.in_proj = nn.ModuleDict({
            ntype: nn.Linear(in_dims_by_type[ntype], hidden)
            for ntype in self.node_types
        })

        # ---- compute standardized edge dim ----
        self.edge_dim_std = max(
            [d for d in edge_dim_by_type.values() if d > 0],
            default=1
        )

        # ---- per-relation edge projections (only if relation has features) ----
        self.edge_proj = nn.ModuleDict({
            str(etype): nn.Linear(edge_dim_by_type[etype], self.edge_dim_std)
            for etype in self.edge_types
            if edge_dim_by_type[etype] > 0
        })

        # ---- mass conservation updates (per node type) ----
        if use_cons_of_mass:
            self.mass_updates = nn.ModuleDict()
            for ntype in self.node_types:
                # Find the self-loop edge type for this node type
                self_edge_str = str((ntype, f"{ntype}edge", ntype))
                
                # Get wl_idx from flux_indices
                wl_idx = flux_indices.get(ntype, {}).get('wl_idx', 0) if flux_indices else 0
                
                # Get slope and length indices
                edge_indices = flux_indices.get(self_edge_str, {}) if flux_indices else {}
                slope_idx = edge_indices.get('slope_idx')
                length_idx = edge_indices.get('length_idx')
                
                mass_update = MassConservationUpdate(
                    wl_idx=wl_idx,
                    edge_slope_idx=slope_idx,
                    edge_length_idx=length_idx,
                    rain_idx=None,
                    dt=1.0,
                    use_learned_flux=True,
                    hidden_flux=32,
                    clamp_nonneg_flux=True,
                )
                self.mass_updates[ntype] = mass_update

        # ---- backbone ----
        backbone = EdgeAwareTransformerBackbone(
            hidden=hidden,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=self.edge_dim_std,
        )

        self.gnn = pyg_nn.to_hetero(backbone, metadata, aggr=aggr)

        # ---- prediction heads (one per node type) ----
        self.heads = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
            for ntype in self.node_types
        })

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, num_timesteps=None, batch=None):
        # NOTE: Mass conservation currently disabled. If re-enabled in future, will need to:
        # 1. Unnormalize x_dict using normalizer.unnormalize() for each feature
        # 2. Run mass conservation in real (unnormalized) space
        # 3. Re-normalize output using normalizer.transform_dynamic()
        # For now, all data is in [0,1] normalized space and model works directly on it.
        
        # project node features (on normalized data)
        x_dict = {
            ntype: F.relu(self.in_proj[ntype](x_dict[ntype]))
            for ntype in self.node_types
        }

        # project / standardize edge features
        out_edge_attr = {}
        for etype in self.edge_types:
            ea = edge_attr_dict[etype]
            key = str(etype)

            if key in self.edge_proj:
                ea = self.edge_proj[key](ea)

            out_edge_attr[etype] = ea

        # If using mass conservation, compute water level updates directly
        if self.use_cons_of_mass and batch is not None:
            # NOTE: Mass conservation is currently disabled in main.py
            # If re-enabled, this code path needs to be updated to work with FeatureNormalizer
            # instead of the old mu/sigma approach
            raise NotImplementedError(
                "Mass conservation mode needs to be updated for FeatureNormalizer. "
                "Please set use_cons_of_mass=False in main.py or update this code path."
            )
        else:
            # Otherwise, use GNN backbone for predictions
            # GNN forward pass (x_dict already projected to hidden)
            batch_dict = None
            if batch is not None:
                batch_dict = {
                    ntype: batch[ntype].batch
                    for ntype in self.node_types
                    if hasattr(batch[ntype], "batch")
                }

            if batch_dict:
                out_x = self.gnn(x_dict, edge_index_dict, out_edge_attr, batch_dict)
            else:
                out_x = self.gnn(x_dict, edge_index_dict, out_edge_attr)
            
            # Prediction heads
            out = {ntype: self.heads[ntype](out_x[ntype]) for ntype in self.node_types}
            
            return out
# ---------------------------------------------------------
# Helper: build edge_attr_dict safely per batch
# ---------------------------------------------------------

def build_edge_attr_dict(batch, edge_dim_std):
    out = {}

    # find dtype/device reference if any edge_attr exists
    ref = None
    for etype in batch.edge_types:
        if "edge_attr" in batch[etype]:
            ref = batch[etype].edge_attr
            break

    dtype = ref.dtype if ref is not None else torch.float32
    device = ref.device if ref is not None else batch[batch.edge_types[0]].edge_index.device

    for etype in batch.edge_types:
        E = batch[etype].edge_index.size(1)

        if "edge_attr" in batch[etype]:
            ea = batch[etype].edge_attr

            # enforce alignment (prevents TransformerConv edge mismatch crash)
            if ea.size(0) != E:
                if ea.size(0) > E:
                    ea = ea[:E]
                else:
                    pad = torch.zeros((E - ea.size(0), ea.size(1)), device=device, dtype=dtype)
                    ea = torch.cat([ea, pad], dim=0)

            out[etype] = ea
        else:
            out[etype] = torch.zeros((E, edge_dim_std), device=device, dtype=dtype)

    return out


# ---------------------------------------------------------
# Model construction from one batch
# ---------------------------------------------------------

def _get_feature_indices(col_names, feature_name):
    """Get index of a feature in column list, or None if not found."""
    try:
        return col_names.index(feature_name)
    except ValueError:
        return None


def build_model_from_batch(batch0, out_dim, edge_col_info=None, water_level_indices=None, use_cons_of_mass=False, norm_stats=None):
    """
    Build model from batch.
    
    Args:
        batch0: first batch from dataloader
        out_dim: output dimension
        edge_col_info: dict with 'edge1_cols' and 'edge2_cols' lists (actual column names)
        water_level_indices: dict with 'oneD' and 'twoD' keys mapping to wl column indices
        use_cons_of_mass: whether to use mass-conservation-based water level updates
    """
    in_dims_by_type = {
        ntype: batch0[ntype].x.size(-1)
        for ntype in batch0.node_types
    }

    edge_dim_by_type = {}
    for etype in batch0.edge_types:
        if "edge_attr" in batch0[etype]:
            edge_dim_by_type[etype] = batch0[etype].edge_attr.size(-1)
        else:
            edge_dim_by_type[etype] = 0

    # Compute feature indices if column info provided
    flux_indices = {}
    if use_cons_of_mass and edge_col_info:
        edge1_cols = edge_col_info.get('edge1_cols', [])
        edge2_cols = edge_col_info.get('edge2_cols', [])
        
        flux_indices = {
            '("oneD", "oneDedge", "oneD")': {
                'slope_idx': _get_feature_indices(edge1_cols, 'slope'),
                'length_idx': _get_feature_indices(edge1_cols, 'length'),
            },
            '("twoD", "twoDedge", "twoD")': {
                'slope_idx': _get_feature_indices(edge2_cols, 'slope'),
                'length_idx': _get_feature_indices(edge2_cols, 'length'),
            },
            '("twoD", "twoDoneD", "oneD")': {
                'slope_idx': _get_feature_indices(edge2_cols, 'slope'),
                'length_idx': _get_feature_indices(edge2_cols, 'length'),
            },
        }
        
        # Store water level indices for mass conservation
        if water_level_indices is None:
            water_level_indices = {'oneD': 0, 'twoD': 0}  # fallback defaults
        
        # Update flux_indices with wl_idx for each node type
        for ntype in ['oneD', 'twoD']:
            wl_idx = water_level_indices.get(ntype, 0)
            flux_indices[ntype] = {'wl_idx': wl_idx}

    model = HeteroEdgeAwareTransformer(
        metadata=batch0.metadata(),
        in_dims_by_type=in_dims_by_type,
        edge_dim_by_type=edge_dim_by_type,
        out_dim=out_dim,
        use_cons_of_mass=use_cons_of_mass,
        flux_indices=flux_indices,
        norm_stats=norm_stats,
    )

    return model

if __name__ == "__main__":
    from data import dl

    batch0 = next(iter(dl))
    model = build_model_from_batch(batch0, 1)
