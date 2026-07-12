import os
os.makedirs('models', exist_ok=True)

from xgb_model import train_xgb_model, load_xgb_model, predict_edge_risks
from data_fetcher import fetch_all

print("Training XGBoost...")
mx = train_xgb_model()
print("XGB RMSE:", mx["rmse"], "R2:", mx["r2"])

data = fetch_all()
xgb = load_xgb_model()
risks, ms = predict_edge_risks(xgb, data["edge_features"])
print("XGB inference:", len(risks), "edges in", ms, "ms")

from gnn_model import train_gnn, load_gnn_model, build_networkx_graph, predict_gnn_risks
print("Training GNN...")
mg = train_gnn()
print("GNN final loss:", mg["loss_history"][-1])

gnn, eo, lh = load_gnn_model()
G = build_networkx_graph(risks)
gr, gms = predict_gnn_risks(gnn, eo, G, risks)
print("GNN inference:", len(gr), "edges in", gms, "ms")
print("ALL OK")
