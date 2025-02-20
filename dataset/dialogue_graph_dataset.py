import os
import random

import torch
import numpy as np
from utils import get_device
from typing import List, Tuple
from itertools import permutations
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from utils.constants import EDGE_TYPES, NEGATIVE_SAMPLES_PERCENTAGE, NUM_RELATIONS


class DialogueGraphDataset(Dataset):

    def __init__(self, root: str, dataset_type: str, k: int = NEGATIVE_SAMPLES_PERCENTAGE, transform=None):
        super().__init__()

        self.root = root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory not found: {root}")

        self.transform = transform
        self.k = k
        self.dataset_type = dataset_type
        self.graph_files = [f for f in os.listdir(root) if f.endswith(".pt")]
        self.device = get_device()

        if len(self.graph_files) == 0:
            raise ValueError(f"No graph found in directory: {root}")

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, index: int) -> HeteroData:

        graph_path = os.path.join(self.root, self.graph_files[index])
        try:
            graph = torch.load(graph_path, map_location=self.device)
        except Exception as e:
            raise ValueError(f"Error loading graph from {graph_path}: {e}")

        relation_types = [edge_type[1] for edge_type in graph.edge_types if
                          edge_type[0] == edge_type[2] and edge_type[0] == 'edu']

        # Collect positive edges
        positive_edges = list()
        relation_labels = []
        for rel_type in relation_types:
            edge_index = graph[rel_type].edge_index
            for src, dst in edge_index.t():
                if src < dst:  # removal of backward arches
                    positive_edges.append((int(src), int(dst)))
                    relation_labels.append(self._encode_relation_type(rel_type))

        num_nodes = graph["edu"].x.size(0)
        all_possible_edges = list(permutations(range(num_nodes), 2))
        negative_edges = [edge for edge in all_possible_edges if edge not in positive_edges]

        # Sample a number of negatives edges
        num_negatives = len(positive_edges) * self.k // 100
        sampled_negative_edges = random.sample(negative_edges, num_negatives)

        # Combine positive and negative pairs
        candidate_edges = list(positive_edges) + sampled_negative_edges
        candidate_edges_index = torch.tensor(np.array(candidate_edges),
                                             dtype=torch.long).T  # shape [2, num_candidate_edges]

        # Labels: 1 for positive, 0 for negative
        candidate_link_labels = torch.tensor(np.array([1] * len(positive_edges) + [0] * len(sampled_negative_edges)),
                                             dtype=torch.long)

        relation_labels = torch.tensor(np.array(relation_labels + [NUM_RELATIONS] * len(sampled_negative_edges)),
                                       dtype=torch.long)

        hetero_data = HeteroData()
        hetero_data["edu"].x = graph["edu"].x
        hetero_data["edu", "to", "edu"].edge_index = candidate_edges_index
        hetero_data["edu", "to", "edu"].link_labels = candidate_link_labels
        hetero_data["edu", "to", "edu"].relation_labels = relation_labels
        hetero_data["edu", "to", "edu"].sampled_negative_edges = torch.tensor(len(sampled_negative_edges))
        hetero_data["edu", "to", "edu"].positive_edges = torch.tensor(len(positive_edges))

        return hetero_data.to(self.device)

    def _generate_labels(self, graph: HeteroData, relation_types: List[str]) -> Tuple[torch.tensor, torch.tensor]:
        num_nodes = graph["edu"].x.size(0)

        # Link prediction labels
        link_labels = torch.zeros((num_nodes, num_nodes), dtype=torch.long)

        # Relation prediction labels
        relation_labels = []

        for rel_type in relation_types:
            edge_index = graph[rel_type].edge_index
            for src, dst in edge_index.t():
                if src < dst:  # removal of backward arches
                    link_labels[src, dst] = 1  # 1 = there is a link
                    relation_labels.append(self._encode_relation_type(rel_type))

        # remove 'self-loops' from link labels
        mask = ~torch.eye(num_nodes, dtype=torch.bool)
        # n x (n-1) matrix -> containing only the link labels for other nodes
        link_labels = link_labels[mask]
        relation_labels = torch.tensor(relation_labels, dtype=torch.long)

        return link_labels, relation_labels

    @staticmethod
    def _encode_relation_type(edge_type) -> int:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown relation type: {edge_type}")

        return EDGE_TYPES.index(edge_type)


if __name__ == '__main__':
    data = DialogueGraphDataset("../data/BALANCED/graphs/test", "train")
    x = data[17]
