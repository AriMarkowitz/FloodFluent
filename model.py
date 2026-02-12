import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


# ---------------------------------------------------------
# Flux Computation (physics-aware message passing)
# ---------------------------------------------------------

class FluxLayer(nn.Module):
    """
    Computes water flux between nodes based on topography.
    Flux = f(water_level_diff, slope, length, [optional: area])
    
    Physical constraints on flux:
    - 'none': allow any flux (positive or negative)
    - 'non_negative': clamp flux >= 0 (only forward flow, no backflow)
    - 'soft_penalty': allow negative but encourage non-negative via regularization
    
    Args:
        hidden: hidden dimension
        slope_idx: column index in edge_attr for slope
        length_idx: column index in edge_attr for length
        use_area: whether to include area in flux computation
        flux_scale: scaling factor for flux output
        flux_constraint: 'none', 'non_negative', or 'soft_penalty'
    """
    def __init__(self, hidden=128, slope_idx=None, length_idx=None, use_area=True, flux_scale=1.0, flux_constraint='none'):
        super().__init__()
        self.slope_idx = slope_idx if slope_idx is not None else 0
        self.length_idx = length_idx if length_idx is not None else 1
        self.use_area = use_area
        self.flux_scale = flux_scale
        self.flux_constraint = flux_constraint
        
        # Learn flux from: [water_level_diff, slope, length] or [water_level_diff, slope, length, area]
        input_dim = 4 if use_area else 3
        self.flux_mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, x, edge_index, edge_attr, node_areas=None):
        """
        Compute flux for each edge.
        
        x: [num_nodes, hidden] - node features
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim] - contains slope and length at specified indices
        node_areas: [num_nodes] - node areas/cross-sections (optional)
        
        Returns: [num_edges, 1] - flux magnitude
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Extract node water levels (use mean across all features)
        x_src = x[src]  # [num_edges, hidden]
        x_dst = x[dst]  # [num_edges, hidden]
        
        # Water level difference drives flux
        h_diff = (x_src - x_dst).mean(dim=1, keepdim=True)  # [num_edges, 1]
        
        # Extract slope and length from edge attributes
        slope = edge_attr[:, self.slope_idx:self.slope_idx+1]  # [num_edges, 1]
        length = edge_attr[:, self.length_idx:self.length_idx+1]  # [num_edges, 1]
        
        # Optionally include area effect on flux magnitude
        if self.use_area:
            if node_areas is None:
                avg_area = torch.zeros_like(h_diff)
            else:
                src_area = node_areas[src].unsqueeze(1)  # [num_edges, 1]
                dst_area = node_areas[dst].unsqueeze(1)  # [num_edges, 1]
                # Use harmonic mean of areas (bottleneck effect)
                avg_area = 2 * src_area * dst_area / (src_area + dst_area + 1e-8)  # [num_edges, 1]
            flux_input = torch.cat([h_diff, slope, length, avg_area], dim=1)  # [num_edges, 4]
        else:
            flux_input = torch.cat([h_diff, slope, length], dim=1)  # [num_edges, 3]
        
        # Learn flux magnitude
        flux = self.flux_mlp(flux_input) * self.flux_scale  # [num_edges, 1]
        
        # Apply physical constraint
        if self.flux_constraint == 'non_negative':
            flux = torch.relu(flux)  # Clamp to >= 0 (no backflow)
        elif self.flux_constraint == 'soft_penalty':
            # Allow negative but penalize it (regularization handled in loss)
            pass
        # else: 'none' - allow any flux
        
        return flux


# ---------------------------------------------------------
# Temporal Attention (over timesteps)
# ---------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Learns to weight the importance of each timestep.
    Input: [num_timesteps, num_nodes, hidden]
    Output: [num_nodes, hidden] (weighted temporal aggregation)
    """
    def __init__(self, hidden=128, heads=4):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads

        self.to_q = nn.Linear(hidden, hidden)
        self.to_k = nn.Linear(hidden, hidden)
        self.to_v = nn.Linear(hidden, hidden)
        self.to_out = nn.Linear(hidden, hidden)
        self.scale = self.head_dim ** -0.5

    def forward(self, x_temporal):
        """
        x_temporal: [timesteps, num_nodes, hidden]
        Returns: [num_nodes, hidden]
        """
        T, N, H = x_temporal.shape

        # Reshape to [T*N, H] for attention computation
        x_flat = x_temporal.reshape(T * N, H)
        
        # Project and reshape: [T*N, H] -> [T*N, heads*head_dim] -> [T, N, heads, head_dim] -> [heads, N, T, head_dim]
        q = self.to_q(x_flat).reshape(T, N, self.heads, self.head_dim).permute(2, 1, 0, 3)  # [heads, N, T, head_dim]
        k = self.to_k(x_flat).reshape(T, N, self.heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.to_v(x_flat).reshape(T, N, self.heads, self.head_dim).permute(2, 1, 0, 3)

        # Attend over timestep dimension: [heads, N, T, T]
        attn = torch.einsum("hntd,hmtd->hnmm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention: [heads, N, T, head_dim]
        out = torch.einsum("hnmm,hmtd->hntd", attn, v)

        # Average over timesteps: [heads, N, head_dim]
        out = out.mean(dim=2)

        # Merge heads and project: [N, H]
        out = out.transpose(0, 1).reshape(N, H)
        out = self.to_out(out)

        return out


# ---------------------------------------------------------
# Homogeneous backbone (edge-feature aware)
# ---------------------------------------------------------

class EdgeAwareTransformerBackbone(nn.Module):
    def __init__(self, hidden=128, heads=4, num_layers=3, dropout=0.3, edge_dim=1):
        super().__init__()

        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.proj  = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                pyg_nn.TransformerConv(
                    in_channels=hidden,
                    out_channels=hidden,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    bias=True,
                )
            )
            self.proj.append(nn.Linear(hidden * heads, hidden))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, x, edge_index, edge_attr):
        for conv, proj, norm in zip(self.convs, self.proj, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = proj(h)
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
        heads=4,
        num_layers=3,
        dropout=0.3,
        aggr="sum",
        use_temporal_attention=True,
        use_flux=False,
        flux_indices=None,
        flux_constraint='none',
    ):
        super().__init__()

        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.use_temporal_attention = use_temporal_attention
        self.use_flux = use_flux

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

        # ---- flux computation (per edge type) ----
        if use_flux:
            self.flux = nn.ModuleDict()
            for etype in self.edge_types:
                etype_str = str(etype)
                # Get indices for this edge type
                indices = flux_indices.get(etype_str, {}) if flux_indices else {}
                slope_idx = indices.get('slope_idx', None)
                length_idx = indices.get('length_idx', None)
                
                flux_layer = FluxLayer(
                    hidden=hidden,
                    slope_idx=slope_idx,
                    length_idx=length_idx,
                    flux_scale=1.0,
                    flux_constraint=flux_constraint,
                )
                self.flux[etype_str] = flux_layer

        # ---- backbone ----
        backbone = EdgeAwareTransformerBackbone(
            hidden=hidden,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=self.edge_dim_std,
        )

        self.gnn = pyg_nn.to_hetero(backbone, metadata, aggr=aggr)

        # ---- temporal attention (per node type) ----
        if use_temporal_attention:
            self.temporal_attn = nn.ModuleDict({
                ntype: TemporalAttention(hidden=hidden, heads=heads)
                for ntype in self.node_types
            })

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
        # Store original x_dict for area extraction before projection
        x_dict_orig = x_dict
        
        # project node features
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

        # Compute and apply flux if enabled
        if self.use_flux and batch is not None:
            for etype in self.edge_types:
                src_type, _, dst_type = etype
                edge_index = edge_index_dict[etype]
                
                # Get node areas from batch metadata if available
                node_areas_src = None
                if "base_area" in batch[src_type]:
                    node_areas_src = batch[src_type].base_area
                elif "cell_area" in batch[src_type]:
                    node_areas_src = batch[src_type].cell_area
                
                # Compute flux for this edge type
                flux = self.flux[str(etype)](
                    x_dict[src_type],
                    edge_index,
                    edge_attr_dict[etype],
                    node_areas=node_areas_src
                )  # [num_edges, 1]
                
                # Scale edge attributes by flux magnitude (gating mechanism)
                flux_gate = torch.sigmoid(flux)  # [num_edges, 1]
                out_edge_attr[etype] = out_edge_attr[etype] * flux_gate
        elif self.use_flux:
            for etype in self.edge_types:
                edge_index = edge_index_dict[etype]
                # Compute flux without area information
                flux = self.flux[str(etype)](
                    x_dict[etype[0]],
                    edge_index,
                    edge_attr_dict[etype],
                    node_areas=None
                )
                flux_gate = torch.sigmoid(flux)
                out_edge_attr[etype] = out_edge_attr[etype] * flux_gate

        h_dict = self.gnn(x_dict, edge_index_dict, out_edge_attr)

        # Apply temporal attention if enabled and num_timesteps provided
        if self.use_temporal_attention and num_timesteps is not None:
            for ntype in self.node_types:
                h = h_dict[ntype]
                # Reshape to [timesteps, num_nodes_per_timestep, hidden]
                num_nodes_per_t = h.size(0) // num_timesteps
                h_temporal = h.reshape(num_timesteps, num_nodes_per_t, -1)
                
                # Apply temporal attention
                h_attn = self.temporal_attn[ntype](h_temporal)
                
                # Replicate across timesteps so we can still use per-timestep masks if needed
                h_dict[ntype] = h_attn.repeat(num_timesteps, 1)

        # return predictions for every node type
        return {
            ntype: self.heads[ntype](h_dict[ntype])
            for ntype in self.node_types
        }


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


def build_model_from_batch(batch0, out_dim, edge_col_info=None, use_flux=True, use_temporal_attention=True, flux_constraint='none'):
    """
    Build model from batch.
    
    Args:
        batch0: first batch from dataloader
        out_dim: output dimension
        edge_col_info: dict with 'edge1_cols' and 'edge2_cols' lists (actual column names)
        use_flux: whether to use flux-based message passing
        use_temporal_attention: whether to use temporal attention
        flux_constraint: 'none', 'non_negative', or 'soft_penalty' - physics constraint on flux
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
    if use_flux and edge_col_info:
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

    model = HeteroEdgeAwareTransformer(
        metadata=batch0.metadata(),
        in_dims_by_type=in_dims_by_type,
        edge_dim_by_type=edge_dim_by_type,
        out_dim=out_dim,
        use_flux=use_flux,
        use_temporal_attention=use_temporal_attention,
        flux_indices=flux_indices,
        flux_constraint=flux_constraint,
    )

    return model

if __name__ == "__main__":
    from data import dl

    batch0 = next(iter(dl))
    model = build_model_from_batch(batch0, 1)
