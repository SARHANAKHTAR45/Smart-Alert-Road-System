"""
gnn_model.py — GraphSAGE GNN Risk Refiner
Hyper-Local Smart Road Hazard Detection & Real-Time Alert System

Stage 2 of the ML pipeline:
  Input:  road network graph + XGBoost base risk scores
  Output: spatially-aware refined risk scores per edge

GraphSAGE learns spatial relationships between adjacent roads —
e.g. if Silk Board floods, nearby Koramangala/HSR roads will
also suffer congestion spikes even without direct hazards.
"""

import os, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from config import EDGES, JUNCTIONS, GNN_MODEL_PATH, GNN_EPOCHS, GNN_LR
from utils import clamp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GraphSAGE Layer (from scratch — no PyG needed)
# ─────────────────────────────────────────────

class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self   = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh  = nn.Linear(in_dim, out_dim, bias=False)
        self.bias       = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, in_dim)  node features
        adj : (N, N)        row-normalised adjacency
        """
        agg  = adj @ x                             # mean-pool neighbours
        out  = self.lin_self(x) + self.lin_neigh(agg) + self.bias
        return F.relu(out)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for edge risk regression."""

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.conv1  = SAGEConv(in_dim, hidden)
        self.conv2  = SAGEConv(hidden, hidden)
        self.head   = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),   # output 0–1, scaled to 0–100 outside
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        h1 = self.dropout(self.conv1(x, adj))
        h2 = self.dropout(self.conv2(h1, adj))
        # Edge representation = concat source + destination embeddings
        edge_h = torch.cat([h2[src_idx], h2[dst_idx]], dim=-1)
        return self.head(edge_h).squeeze(-1)   # (E,)


# ─────────────────────────────────────────────
# Graph builder helpers
# ─────────────────────────────────────────────

def build_networkx_graph(edge_risks: dict | None = None) -> nx.DiGraph:
    G = nx.DiGraph()
    for jid, info in JUNCTIONS.items():
        G.add_node(jid, **info)
    for (src, dst, road_type, length_km) in EDGES:
        risk = edge_risks.get((src, dst), 50.0) if edge_risks else 50.0
        G.add_edge(src, dst, road_type=road_type, length_km=length_km, risk=risk)
    return G

def update_graph_risks(G: nx.DiGraph, gnn_risks: dict) -> None:
    for (src, dst), risk in gnn_risks.items():
        if G.has_edge(src, dst):
            G[src][dst]["risk"] = risk

def graph_to_edge_table(G: nx.DiGraph) -> list[dict]:
    rows = []
    for src, dst, data in G.edges(data=True):
        rows.append({
            "Edge":     f"{JUNCTIONS[src]['name']} → {JUNCTIONS[dst]['name']}",
            "Risk":     round(data.get("risk", 0), 1),
            "Length km": data.get("length_km", 0),
            "Road Type": ["Arterial","Highway","Local"][data.get("road_type", 0)],
        })
    return rows

def _build_torch_data(G: nx.DiGraph, xgb_risks: dict, edge_order: list):
    N = G.number_of_nodes()
    nodes = sorted(G.nodes())
    nid2i = {n: i for i, n in enumerate(nodes)}

    # Node features: lat, lon, is_major, acc_score, avg_xgb_risk
    x = []
    for nid in nodes:
        info  = JUNCTIONS[nid]
        lat   = (info["lat"] - 12.9) / 0.3
        lon   = (info["lon"] - 77.5) / 0.3
        major = float(info["is_major"])
        from config import HISTORICAL_ACCIDENTS
        acc   = HISTORICAL_ACCIDENTS.get(nid, 3) / 10.0
        out_e = [e for e in edge_order if e[0] == nid]
        avg_r = np.mean([xgb_risks.get(e, 50) for e in out_e]) / 100.0 if out_e else 0.5
        x.append([lat, lon, major, acc, avg_r])

    x_t = torch.tensor(x, dtype=torch.float32)

    # Row-normalised adjacency
    adj = torch.zeros(N, N)
    for src, dst in G.edges():
        i, j = nid2i[src], nid2i[dst]
        adj[i, j] = 1.0
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1)
    adj = adj / row_sum

    src_idx = torch.tensor([nid2i[e[0]] for e in edge_order], dtype=torch.long)
    dst_idx = torch.tensor([nid2i[e[1]] for e in edge_order], dtype=torch.long)
    y       = torch.tensor([xgb_risks.get(e, 50.0) / 100.0 for e in edge_order], dtype=torch.float32)

    return x_t, adj, src_idx, dst_idx, y


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_gnn() -> dict:
    os.makedirs(os.path.dirname(GNN_MODEL_PATH), exist_ok=True)
    logger.info("Training GraphSAGE GNN…")

    from xgb_model import load_xgb_model, predict_edge_risks
    from data_fetcher import fetch_all

    # Use a dry-run fetch to get synthetic edge features
    pipeline_data  = fetch_all()
    xgb_model      = load_xgb_model()
    xgb_risks, _   = predict_edge_risks(xgb_model, pipeline_data["edge_features"])

    G          = build_networkx_graph(xgb_risks)
    edge_order = list(G.edges())

    x_t, adj, src_idx, dst_idx, y = _build_torch_data(G, xgb_risks, edge_order)

    model     = GraphSAGE(in_dim=x_t.shape[1], hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(GNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(x_t, adj, src_idx, dst_idx)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(round(float(loss.item()), 6))

    torch.save({
        "model_state":  model.state_dict(),
        "in_dim":       x_t.shape[1],
        "edge_order":   edge_order,
        "loss_history": loss_history,
    }, GNN_MODEL_PATH)

    logger.info(f"GNN trained: final loss = {loss_history[-1]}")
    return {"loss_history": loss_history}


# ─────────────────────────────────────────────
# Load & Inference
# ─────────────────────────────────────────────

def load_gnn_model() -> tuple:
    ckpt       = torch.load(GNN_MODEL_PATH, map_location="cpu", weights_only=False)
    model      = GraphSAGE(in_dim=ckpt["in_dim"], hidden=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["edge_order"], ckpt.get("loss_history", [])


def predict_gnn_risks(
    model: GraphSAGE,
    edge_order: list,
    G: nx.DiGraph,
    xgb_risks: dict,
) -> tuple[dict, float]:
    x_t, adj, src_idx, dst_idx, _ = _build_torch_data(G, xgb_risks, edge_order)

    t0 = time.perf_counter()
    with torch.no_grad():
        model.eval()
        preds = model(x_t, adj, src_idx, dst_idx).numpy()
    infer_ms = (time.perf_counter() - t0) * 1000

    # Blend: 40% GNN refinement + 60% XGBoost baseline
    gnn_risks = {}
    for edge, gnn_pred in zip(edge_order, preds):
        xgb_val = xgb_risks.get(edge, 50.0)
        gnn_val = float(gnn_pred) * 100.0
        blended = clamp(0.6 * xgb_val + 0.4 * gnn_val, 0, 100)
        gnn_risks[edge] = round(blended, 2)

    return gnn_risks, round(infer_ms, 2)


# ─────────────────────────────────────────────
# Pure ML Greedy Routing  (no Dijkstra)
# ─────────────────────────────────────────────

def find_ml_path(
    G: nx.DiGraph,
    gnn_risks: dict,
    src_id: int,
    dst_id: int,
    blocked_edges: set | None = None,
) -> tuple[list, float]:
    """
    Greedy ML traversal: at each node, choose the neighbour with
    the lowest GNN-predicted risk on the outgoing edge.

    No Dijkstra. No Bellman-Ford. Pure ML-guided greedy traversal.
    """
    blocked = blocked_edges or set()
    visited = {src_id}
    path    = [src_id]
    total_r = 0.0

    MAX_HOPS = len(JUNCTIONS) + 2

    for _ in range(MAX_HOPS):
        curr = path[-1]
        if curr == dst_id:
            break

        neighbours = [
            (dst, gnn_risks.get((curr, dst), 100.0))
            for dst in G.successors(curr)
            if dst not in visited and (curr, dst) not in blocked
        ]
        if not neighbours:
            # Dead end — backtrack is not supported in greedy; return failure
            return [], float("inf")

        # Pick least-risk unvisited neighbour
        nxt, risk = min(neighbours, key=lambda x: x[1])
        path.append(nxt)
        visited.add(nxt)
        total_r += risk

    if path[-1] != dst_id:
        return [], float("inf")

    return path, round(total_r, 1)
