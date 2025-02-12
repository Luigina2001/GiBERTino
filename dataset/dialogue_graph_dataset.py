import os
from itertools import permutations

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any
from torch_geometric.data import HeteroData
import numpy as np

from utils import get_device


class DialogueGraphDataset(Dataset):

    def __init__(self, root: str, dataset_type: str, transform=None):
        super().__init__()

        self.root = root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory not found: {root}")

        self.transform = transform
        self.dataset_type = dataset_type
        self.graph_files = [f for f in os.listdir(root) if f.endswith(".pt")]
        self.device = get_device()

        if len(self.graph_files) == 0:
            raise ValueError(f"No graph found in directory: {root}")

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        graph_path = os.path.join(self.root, self.graph_files[index])
        try:
            graph = torch.load(graph_path, map_location=self.deviceli)
        except Exception as e:
            raise ValueError(f"Error loading graph from {graph_path}: {e}")

        relation_types = [key for key in graph.keys() if isinstance(key, tuple)
                          and key[0] == "edu" and key[2] == "edu"]

        link_labels, relation_labels = self._generate_labels(graph, relation_types)

        edges_permutations = list(permutations(torch.arange(graph["edu"].x.size(0)).tolist(), 2))

        return {
            "x": graph["edu"].x,
            "edge_indices": torch.tensor(np.array(edges_permutations), dtype=torch.long).T,
            "link_labels": link_labels,
            "relation_labels": relation_labels
        }

    def _generate_labels(self, graph: HeteroData, relation_types: List[str]) -> Tuple[torch.tensor, torch.tensor]:
        num_nodes = graph["edu"].x.size(0)

        # Link prediction labels
        link_labels = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

        # Relation prediction labels
        relation_labels = []

        for rel_type in relation_types:
            edge_index = graph[rel_type].edge_index
            for src, dst in edge_index.t():
                link_labels[src, dst] = 1  # 1 = there is a link
                relation_labels.append(self._encode_relation_type(rel_type[1]))

        # remove 'self-loops' from link labels
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        # n x (n-1) matrix -> containing only the link labels for other nodes
        link_labels = link_labels[mask].view(num_nodes, -1)

        relation_labels = torch.tensor(relation_labels, dtype=torch.long)

        return link_labels, relation_labels

    @staticmethod
    def _encode_relation_type(edge_type) -> int:
        edge_types = ['Clarification_question', 'Explanation', 'Contrast', 'Comment',
                      'Elaboration', 'Result', 'QAP', 'Correction', 'Narration',
                      'Acknowledgement', 'Q-Elab', 'Continuation']

        if edge_type not in edge_types:
            raise ValueError(f"Unknown relation type: {edge_type}")

        return edge_types.index(edge_type)
