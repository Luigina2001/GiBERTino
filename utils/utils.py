import json
import random
import logging
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import torch.backends.mps
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_dataset(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)


def display_graph(graph: HeteroData, dataset_name: Optional[str] = None):
    try:
        # Convert the graph to a NetworkX graph for visualization
        G = to_networkx(graph)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(
            G, seed=42
        )  # Layout for consistent visualization

        # Get unique edge types
        edge_types = set(
            data["type"][1] for _, _, data in G.edges(data=True)
        )
        colors = list(mcolors.TABLEAU_COLORS.values())
        random.shuffle(colors)  # Shuffle to assign unique colors
        color_map = {
            etype: colors[i % len(colors)]
            for i, etype in enumerate(edge_types)
        }

        # Draw edges with unique colors based on relation type
        legend_handles = []
        for edge_type in edge_types:
            edges = [
                (u, v)
                for u, v, data in G.edges(data=True)
                if data["type"][1] == edge_type
            ]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color_map[edge_type],
                width=2,
                alpha=0.7,
                label=edge_type,
            )
            legend_handles.append(
                mpatches.Patch(color=color_map[edge_type], label=edge_type)
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=300,
            node_color="lightblue",
            edgecolors="black",
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

        plt.legend(
            handles=legend_handles,
            title="Edge Types",
            loc="upper right",
            bbox_to_anchor=(1, 1),
        )
        title = "Graph Visualization"
        if dataset_name is not None:
            title += f": {dataset_name}"

        plt.title(title)
        plt.show()

    except Exception as e:
        logger.warning(f"Failed to display graph: {e}.")


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def print_metrics(step, metrics):
    table = Table(title=f"Metrics at step {step}")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console = Console()
    console.print(table)
