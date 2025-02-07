import logging
import os
import random
from collections import defaultdict

import numpy as np
from typing import Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import torch
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
from transformers import pipeline
import transformers.utils.logging

from utils import load_dataset
from utils.constants import SENTIMENTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TODO: disable huggingface logger because this does not work
transformers.utils.logging.set_verbosity_error()  # disable info and warning logging (progress bar and updates)


class GraphBuilder:
    def __init__(self, dataset_path: str, dataset_name: Literal['MOLWENI', 'STAC', 'MINECRAFT'],
                 sentence_model: str = 'Alibaba-NLP/gte-modernbert-base',
                 sentiment_model: str = 'finiteautomata/bertweet-base-sentiment-analysis'):
        """
        Initialize the GraphBuilder with the dataset path and name.

        Args:
            dataset_path (str): Path to the dataset.
            dataset_name (Literal['MOLWENI', 'STAC', 'MINECRAFT']): Name of the dataset.
        """
        logger.info(f"Loading dataset from {dataset_path}")
        self.dialogs = load_dataset(dataset_path)
        logger.info(f"{dataset_name} dataset loaded successfully.")

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.sentence_model = SentenceTransformer(sentence_model)
        self.sentiment_model = pipeline('sentiment-analysis', model=sentiment_model)
        self.graphs = []

    def __call__(self):
        """
        Build a heterogeneous graph from the dataset where edges represent relationships between EDUs.
        """
        logger.info(f"Starting graph construction for dataset {self.dataset_name}...")

        for dialog in tqdm(self.dialogs, desc="Processing dialogs"):
            graph_data = HeteroData()
            node_idxs = []
            text_embeddings = []
            sentiment_labels = []
            speakers = {}
            speakers_ids = []

            for edu_idx, edu in enumerate(dialog["edus"]):
                text_embeddings.append(self.sentence_model.encode(edu["text"]))
                sentiment = self.sentiment_model(edu["text"])[0]['label']
                sentiment_labels.append(SENTIMENTS[sentiment])

                if edu["speaker"] not in speakers:
                    speakers[edu["speaker"]] = len(speakers)

                speakers_ids.append(speakers[edu["speaker"]])
                node_idxs.append(edu_idx)

            embeddings_dim = text_embeddings[0].shape[-1]
            text_embeddings = torch.tensor(np.array(text_embeddings), dtype=torch.double)
            sentiment_labels = torch.tensor(np.array(sentiment_labels), dtype=torch.int8)
            speakers_ids = torch.tensor(np.array(speakers_ids), dtype=torch.int8)

            # match dim 1
            sentiment_labels = sentiment_labels.unsqueeze(1).repeat(1, embeddings_dim)
            speakers_ids = speakers_ids.unsqueeze(1).repeat(1, embeddings_dim)

            node_features = torch.cat([text_embeddings, sentiment_labels, speakers_ids], dim=-1)

            graph_data["edu"].x = node_features

            edge_dict = defaultdict(list)

            for rel_idx in range(len(dialog["relations"])):
                src = dialog["relations"][rel_idx]["x"]
                dst = dialog["relations"][rel_idx]["y"]
                rel_type = dialog["relations"][rel_idx]["type"]

                edge_dict[rel_type].append((src, dst))

            for rel_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                graph_data["edu", rel_type, "edu"].edge_index = edge_tensor

            self.graphs.append(graph_data)

        logger.info("Graph construction completed successfully.")

    def save_graphs(self, path: str, graph: Optional[HeteroData]):
        """
        Save the constructed graph to a file.

        Args:
            path (str): Path where the graphs will be saved.
            graph: Specific graph to save. Optional.
        """
        if len(self.graphs) == 0 and graph is None:
            logger.warning("No graph to save. Please run the graph construction first.")
            return

        if graph is None and not os.path.isdir(path):
            logger.warning("Path is not a directory. Please specify a directory")
            return

        logger.info(f"Saving graphs to {path}")

        if graph is None:
            os.makedirs(path, exist_ok=True)

            for idx, graph in enumerate(self.graphs):
                graph_path = os.path.join(path, str(idx))
                torch.save(graph, os.path.join(graph_path, ".pt"))
        else:
            torch.save(graph, path)
        logger.info("Graphs saved successfully.")

    def display_graph(self, idx: int):
        """
        Display the graph using networkx and matplotlib. If display is not possible,
        save the graph to a file and log the path.
        """
        if len(self.graphs) == 0:
            logger.warning("No graph to display. Please run the graph construction first.")
            return

        try:
            # Convert the graph to a NetworkX graph for visualization
            G = to_networkx(self.graphs[idx])

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
            self.save_graphs(save_path, graph=self.graphs[idx])
            logger.info(f"Graph saved to {save_path} for manual inspection.")


if __name__ == '__main__':
    builder = GraphBuilder(dataset_path='../data/MOLWENI/test.json', dataset_name='MOLWENI')
    builder()
