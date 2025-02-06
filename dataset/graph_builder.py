import torch
import numpy as np
from torch_geometric.data import HeteroData
from dataset.preprocessing import extract_features


def build_graph(dialog):
    data = HeteroData()

    node_idx = {edu["speechturn"]: idx for idx, edu in enumerate(dialog["edus"])}

    embeddings, sentiments, speakers = extract_features(dialog["edus"])

    data["edu"].x = torch.tensor(np.array(embeddings), dtype=torch.float)
    data["edu"].sentiment = torch.tensor(sentiments, dtype=torch.float)

    speaker_to_id = {s: i for i, s in enumerate(set(speakers))}
    data["edu"].speaker = torch.tensor([speaker_to_id[s] for s in speakers], dtype=torch.long)

    edge_index = {}

    for rel in dialog["relations"]:
        src = node_idx.get(rel["x"])
        dst = node_idx.get(rel["y"])
        if src is not None and dst is not None:
            if rel["type"] not in edge_index:
                edge_index[rel["type"]] = []
            edge_index[rel["type"]].append([src, dst])

    for rel_type, edges in edge_index.items():
        if edges:
            data['edu', rel_type, 'edu'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return data
