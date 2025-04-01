import json
import random
import logging
import os
import imageio.v2 as imageio
from typing import Optional, List
import shutil
import uuid

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import torch.backends.mps
from matplotlib import pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import to_networkx
from rich.console import Console
from rich.table import Table

from utils.constants import RELATIONS_COLOR_MAPS

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


def create_graph_from_predictions(graph: HeteroData, relations: List[str], preds: torch.Tensor,
                                  display_graphs: bool = True) -> HeteroData:
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    new_graph = HeteroData()
    new_graph["edu"].x = graph["edu"].x
    new_graph["edu"].edus = graph["edu"].edus

    edge_index = graph["edu", "to", "edu"].edge_index.T[preds == 1]
    rel_labels = graph["edu", "to", "edu"].rel_labels[preds == 1]

    for rel_type in rel_labels:
        new_graph["edu", relations[rel_type], "edu"].edge_index = edge_index[
            (rel_labels == rel_type).nonzero(as_tuple=True)[0].tolist()].T.contiguous()  # noqa

    graph_rel_labels = graph["edu", "to", "edu"].rel_labels
    graph_edge_index = graph["edu", "to", "edu"].edge_index.T
    del graph["edu", "to", "edu"]

    for rel_type in graph_rel_labels:
        graph["edu", relations[rel_type], "edu"].edge_index = graph_edge_index[
            (rel_labels == rel_type).nonzero(as_tuple=True)[0].tolist()].T.contiguous()  # noqa

    if display_graphs:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        plt.sca(axs[0])
        display_graph(graph, display_legend=False, ax=axs[0])
        axs[0].set_title("Original Discourse Graph")

        plt.sca(axs[1])
        display_graph(new_graph, ax=axs[1])
        axs[1].set_title("Predicted Discourse Graph")

    return new_graph


def create_gif_from_graphs(g: nx, new_graph: nx, all_paths: List, filename: str = "graph_evolution.gif"):
    temp_dir = "temp_frames"
    frames = []

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Convert input graphs to NetworkX format
    G_real = to_networkx(g, to_undirected=False)
    G_pred = to_networkx(new_graph, to_undirected=False)

    pos = nx.spring_layout(G_real, seed=42)

    plt.rcParams.update({
        'font.size': 12,
        'axes.edgecolor': '#cccccc',
        'axes.labelcolor': '#666666',
        'figure.facecolor': 'white',
        'savefig.transparent': False
    })

    NODE_SIZE = 800
    EDGE_WIDTH = 1.5
    HIGHLIGHT_WIDTH = 3
    COLOR_PALETTE = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#f4a261', '#e76f51', '#2a9d8f', '#264653']

    def draw_graph(G, highlighted_edges=[], title="", path_num=0):  # noqa
        plt.figure(figsize=(10, 8))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='#66b3ff',
                               node_size=NODE_SIZE, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        # Draw normal edges
        nx.draw_networkx_edges(G, pos, width=EDGE_WIDTH,
                               edge_color='#cccccc', arrows=True)

        # Highlight specific edges
        if highlighted_edges:
            edge_color = COLOR_PALETTE[path_num % len(COLOR_PALETTE)]
            nx.draw_networkx_edges(
                G, pos, edgelist=highlighted_edges,
                edge_color=edge_color, width=HIGHLIGHT_WIDTH,
                arrows=True, arrowstyle='-|>', arrowsize=15
            )

        # Add title and formatting
        plt.title(f"{title}\n" + (f"Path {path_num + 1}" if path_num else ""),
                  fontsize=14, pad=20)
        plt.axis('off')

        # Save frame with a unique name
        frame_path = os.path.join(temp_dir, f"frame_{uuid.uuid4().hex}.png")
        plt.savefig(frame_path, bbox_inches='tight', dpi=100)
        plt.close()
        return frame_path

    # Initial frames showing both graphs
    frames.append(draw_graph(G_real, title="Original Graph"))
    frames.append(draw_graph(G_pred, title="Predicted Graph"))

    # Progressive path construction
    for path_num, path in enumerate(all_paths):
        current_path_edges = []
        for i in range(len(path) - 1):
            current_edge = (path[i], path[i + 1])
            current_path_edges.append(current_edge)

            frame = draw_graph(
                G_pred,
                highlighted_edges=current_path_edges,  # Show only the current path
                title="Building Paths",
                path_num=path_num
            )
            frames.append(frame)

    durations = [100.0, 100.0] + [4.0] * (len(frames) - 3) + [10.0] * len(all_paths)

    # Create the GIF
    with imageio.get_writer(filename, mode='I', duration=durations) as writer:
        for frame in frames:
            try:
                writer.append_data(imageio.imread(frame))
            except Exception as e:
                print(f"Error processing {frame}: {str(e)}")

    # Clean up temporary frames
    for frame in frames:
        try:
            os.remove(frame)
        except FileNotFoundError:
            continue

    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            shutil.rmtree(temp_dir)

    print(f"GIF successfully created: {filename}")
    return filename


def display_graph(graph: HeteroData, dataset_name: Optional[str] = None, display_legend: bool = True,
                  ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    try:
        # Convert to NetworkX graph for visualization
        G = to_networkx(graph)

        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()  # Get current axis if none provided

            if title is None:
                title = "Graph Visualization"

                if dataset_name:
                    title += f": {dataset_name}"

            ax.set_title(title)

        ax.clear()

        pos = nx.spring_layout(G, seed=42)  # Consistent layout

        # Get unique edge types
        edge_types = set(data["type"][1] for _, _, data in G.edges(data=True))

        # Generate random colors for any edge type missing from RELATIONS_COLOR_MAPS
        available_colors = list(mcolors.TABLEAU_COLORS.values())
        random.shuffle(available_colors)
        fallback_colors = iter(available_colors)

        color_map = {
            etype: RELATIONS_COLOR_MAPS.get(etype, next(fallback_colors))
            for etype in edge_types
        }

        # Draw edges with assigned colors
        legend_handles = []
        for edge_type, color in color_map.items():
            edges = [(u, v) for u, v, data in G.edges(data=True) if data["type"][1] == edge_type]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, width=2, alpha=0.7)
            legend_handles.append(mpatches.Patch(color=color, label=edge_type))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue", edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

        if display_legend:
            # Add legend
            ax.legend(handles=legend_handles, title="Edge Types", loc="upper left", bbox_to_anchor=(1.05, 1),
                      borderaxespad=0.)

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
