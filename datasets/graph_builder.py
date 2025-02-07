import logging
import os
import random
from typing import Literal

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import matplotlib.patches as mpatches
from datasets.preprocessing import extract_features, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, dataset_path: str, dataset_name: Literal['MOLWENI', 'STAC', 'MINECRAFT']):
        """
        Initialize the GraphBuilder with the dataset path and name.

        Args:
            dataset_path (str): Path to the dataset.
            dataset_name (Literal['MOLWENI', 'STAC', 'MINECRAFT']): Name of the dataset.
        """
        logger.info(f"Loading dataset from {dataset_path}")

        self.dialogs = load_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        logger.info(f"{dataset_name} dataset loaded successfully.")

    def __call__(self):
        """
        Build a heterogeneous graph from the dataset where edges represent relationships between EDUs.
        """

        logger.info(f"Starting graph construction for dataset {self.dataset_name}...")
        # Use tqdm to show progress for dialog processing
        for dialog in tqdm(self.dialogs, desc="Processing dialogs"):
            graph_data = HeteroData()
            node_idxs = []

            for edu_idx in range(len(dialog["edus"])):
                node_idxs.append(edu_idx)

            embeddings, sentiments, speakers = extract_features(dialog["edus"])

            speaker_to_id = [i for i, s in enumerate(set(speakers))]
            node_ids = torch.tensor(np.array(speaker_to_id), dtype=torch.float)
            text_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float)
            sentiment_labels = torch.tensor(sentiments, dtype=torch.float)

            node_features = torch.cat([node_ids, text_embeddings, sentiment_labels], dim=-1)
            graph_data["edu"].x = node_features

            for rel_idx in range(len(dialog["relations"])):
                src = dialog["relations"][rel_idx]["x"]
                dst = dialog["relations"][rel_idx]["y"]
                rel_type = dialog["relations"][rel_idx]["type"]

                graph_data["edu", rel_type, "edu"].edge_index = (torch.tensor([[src, dst]]).t().contiguous())

        self.graph = graph_data
        print(dialog["id"])
        logger.info("Graph construction completed successfully.")

    def save_graph(self, path: str):
        """
        Save the constructed graph to a file.

        Args:
            path (str): Path where the graph will be saved.
        """
        if self.graph is None:
            logger.warning("No graph to save. Please run the graph construction first.")
            return

        logger.info(f"Saving graph to {path}")
        torch.save(self.graph, path)
        logger.info("Graph saved successfully.")

    def display_graph(self):
        """
        Display the graph using networkx and matplotlib. If display is not possible,
        save the graph to a file and log the path.
        """
        if self.graph is None:
            logger.warning("No graph to display. Please run the graph construction first.")
            return

        try:
            # Convert the graph to a NetworkX graph for visualization
            G = to_networkx(self.graph)

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)  # Layout for consistent visualization

            # Get unique edge types
            edge_types = set(data['type'][1] for _, _, data in G.edges(data=True))
            colors = list(mcolors.TABLEAU_COLORS.values())
            random.shuffle(colors)  # Shuffle to assign unique colors
            color_map = {etype: colors[i % len(colors)] for i, etype in enumerate(edge_types)}

            # Draw edges with unique colors based on relation type
            legend_handles = []
            for edge_type in edge_types:
                edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'][1] == edge_type]
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color_map[edge_type], width=2, alpha=0.7,
                                       label=edge_type)
                legend_handles.append(mpatches.Patch(color=color_map[edge_type], label=edge_type))

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', edgecolors='black')
            nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

            plt.legend(handles=legend_handles, title="Edge Types", loc="upper right", bbox_to_anchor=(1, 1))
            plt.title(f"Graph Visualization: {self.dataset_name}")
            plt.show()

        except Exception as e:
            logger.warning(f"Graph display failed: {e}. Saving graph to a file instead.")
            save_path = os.path.join(os.getcwd(), f"{self.dataset_name}_graph.pth")
            self.save_graph(save_path)
            logger.info(f"Graph saved to {save_path} for manual inspection.")


def build_graph(dialog):
    data = HeteroData()

    node_idx = {
        edu["speechturn"]: idx for idx, edu in enumerate(dialog["edus"])
    }

    embeddings, sentiments, speakers = extract_features(dialog["edus"])

    data["edu"].x = torch.tensor(np.array(embeddings), dtype=torch.float)
    data["edu"].sentiment = torch.tensor(sentiments, dtype=torch.float)

    speaker_to_id = {s: i for i, s in enumerate(set(speakers))}
    data["edu"].speaker = torch.tensor(
        [speaker_to_id[s] for s in speakers], dtype=torch.long
    )

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
            data["edu", rel_type, "edu"].edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
            )

    return data
