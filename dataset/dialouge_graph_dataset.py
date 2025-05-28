import os
import random
from itertools import permutations

import numpy as np
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

from utils import get_device
from utils.constants import NEGATIVE_SAMPLES_RATIO, RELATIONS


class DialogueGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        negative_sampling_ratio: float = NEGATIVE_SAMPLES_RATIO,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.device = get_device()
        self.relations = RELATIONS["UNIFIED"]
        self.relations.append("Unknown")
        self.negative_sampling_ratio = negative_sampling_ratio

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False, map_location=self.device
        )

    @property
    def processed_file_names(self):
        return ["processed_data.pt"]

    @property
    def raw_file_names(self):
        return [
            filename for filename in os.listdir(self.root) if filename.endswith(".pt")
        ]

    def process(self):
        data_list = []

        for graph_filename in os.listdir(self.root):
            if not graph_filename.endswith(".pt"):
                continue

            try:
                graph = torch.load(
                    os.path.join(self.root, graph_filename),
                    map_location=self.device,
                    weights_only=False,
                )
            except Exception as e:
                raise ValueError(f"Error loading graph from {graph_filename}: {e}")

            # collect positive edges
            pos_edges = []
            rel_labels = []
            for edge_type in graph.edge_types:
                graph_edge_index = graph[edge_type[1]].edge_index
                pos_edges.extend(graph_edge_index.t().tolist())
                rel_labels.extend(
                    [self.encode_relation_type(edge_type[1])]
                    * graph_edge_index.shape[1]
                )

            # generate all possible edges and sample negative edges
            num_nodes = graph["edu"].x.shape[0]
            all_possible_edges = list(permutations(range(num_nodes), 2))
            neg_edges = [edge for edge in all_possible_edges if edge not in pos_edges]

            num_negatives = int(len(pos_edges) * self.negative_sampling_ratio / 100)
            neg_edges = random.sample(neg_edges, num_negatives)

            # combine positive and negative edges
            edges = np.array(pos_edges + neg_edges)
            edge_index = torch.tensor(np.array(edges), dtype=torch.long).T
            # label: 1 for pos, 0 for neg
            link_labels = torch.tensor(
                np.array([1] * len(pos_edges) + [0] * len(neg_edges)), dtype=torch.long
            )
            # label: [0-11] for pos, 12 for neg
            rel_labels = torch.tensor(
                np.array(
                    rel_labels + [self.encode_relation_type("Unknown")] * len(neg_edges)
                ),
                dtype=torch.long,
            )

            new_graph = HeteroData()
            new_graph["edu"].x = graph["edu"].x
            new_graph["edu"].edus = graph["edu"].edus
            # Structural edges (edges that define the graph)
            new_graph["edu", "to", "edu"].edge_index = edge_index

            # Training edges (link prediction task)
            new_graph["edu", "to", "edu"].link_labels = link_labels
            new_graph["edu", "to", "edu"].rel_labels = rel_labels

            data_list.append(new_graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def encode_relation_type(self, edge_type: str) -> int:  # noqa
        if edge_type not in self.relations:
            raise ValueError(f"Unknown relation type: {edge_type}")
        return self.relations.index(edge_type)
