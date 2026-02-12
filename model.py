import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


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
    ):
        super().__init__()

        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])

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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
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

        h_dict = self.gnn(x_dict, edge_index_dict, out_edge_attr)

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

def build_model_from_batch(batch0, out_dim):
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

    model = HeteroEdgeAwareTransformer(
        metadata=batch0.metadata(),
        in_dims_by_type=in_dims_by_type,
        edge_dim_by_type=edge_dim_by_type,
        out_dim=out_dim,
    )

    return model

if __name__ == "__main__":
    from data import dl

    batch0 = next(iter(dl))
    model = build_model_from_batch(batch0, 1)
