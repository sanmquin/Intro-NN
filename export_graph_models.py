import os
import json
import torch

# Ensure destination directory exists
out_dir = "web_graph/src/data"
os.makedirs(out_dir, exist_ok=True)

# 1. Load checkpoints
c300_path = "graphs/data/ar_graph_transformer_epoch_300.pt"
c400_path = "graphs/data/ar_graph_transformer_epoch_400.pt"

c300 = torch.load(c300_path, map_location="cpu")
c400 = torch.load(c400_path, map_location="cpu")

def export_state_dict(sd):
    exported = {}
    for k, v in sd.items():
        exported[k] = v.detach().cpu().numpy().tolist()
    return exported

weights_payload = {
    "300": export_state_dict(c300["model_state_dict"]),
    "400": export_state_dict(c400["model_state_dict"]),
    "config": c300.get("config", {})
}

with open(os.path.join(out_dir, "models_weights.json"), "w") as f:
    json.dump(weights_payload, f)

print(f"Exported models_weights.json with {len(weights_payload['300'])} parameters per checkpoint.")

# 2. Export training history
history_payload = {
    "300": c300.get("history", {}),
    "400": c400.get("history", {})
}

with open(os.path.join(out_dir, "training_history.json"), "w") as f:
    json.dump(history_payload, f)

print("Exported training_history.json.")

# 3. Export graph test samples
dataset_path = "graphs/graphs/data/graph_dfs_dataset.pt"
dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
test_raw = dataset["test"]

samples = []
for idx in range(min(20, len(test_raw))):
    item = test_raw[idx]
    trace, sp, G, mapping = item[0], item[1], item[2], item[3]
    backtracks = item[4] if len(item) > 4 else 0
    node_backtraces = item[5] if len(item) > 5 else {}

    nodes = sorted(list(G.nodes()))
    edges = [list(e) for e in G.edges()]

    # Format node positions using spring layout for reproducible 2D node placement in graph visualizer
    import networkx as nx
    pos = nx.spring_layout(G, seed=42 + idx)
    node_coords = {str(node): [float(pos[node][0]), float(pos[node][1])] for node in nodes}

    samples.append({
        "id": idx,
        "trace": trace,
        "sp": sp,
        "nodes": nodes,
        "edges": edges,
        "node_coords": node_coords,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "backtracks": backtracks,
        "node_backtraces": {str(k): int(v) for k, v in node_backtraces.items()} if isinstance(node_backtraces, dict) else {}
    })

samples_payload = {
    "vocab_size": dataset.get("vocab_size", 42),
    "pad_token": dataset.get("pad_token", 40),
    "stop_token": dataset.get("stop_token", 41),
    "max_src_len": dataset.get("max_src_len", 50),
    "max_tgt_len": dataset.get("max_tgt_len", 21),
    "samples": samples
}

with open(os.path.join(out_dir, "graph_samples.json"), "w") as f:
    json.dump(samples_payload, f)

print(f"Exported graph_samples.json with {len(samples)} test samples.")
