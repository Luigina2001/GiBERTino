import logging
import os
from collections import defaultdict
from typing import Literal, Optional, Union, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

from transformations import BackTranslation
from utils import load_dataset, display_graph, get_device
from utils.constants import SENTIMENTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 8


class GraphBuilder:
    def __init__(
            self,
            dataset_paths: Union[List[str], str],
            dataset_names: Union[str, List[str]],
            dataset_type: Literal["test", "train", "val"],
            sentence_model: str = "Alibaba-NLP/gte-modernbert-base",
            sentiment_model: str = "finiteautomata/bertweet-base-sentiment-analysis",
            src_translator: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
            tgt_translator: str = "Helsinki-NLP/opus-mt-ROMANCE-en",
            p: float = 0.5,
            augment_data: bool = True
    ):
        """
        Initialize the GraphBuilder with the dataset path and name.

        Args:
            dataset_paths (Union[List[str], str]): Path(s) to the dataset(s).
            dataset_names (Union[str, List[str]]): Name(s) of the dataset(s).
            dataset_type (Literal['test', 'train', 'val', 'dev']): Type of dataset.
            sentence_model (str): Name of the sentence embedding model.
            sentiment_model (str): Name of the sentiment analysis model.
        """
        # Ensure dataset_paths and dataset_names are lists
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        if len(dataset_paths) != len(dataset_names):
            raise ValueError(
                "The number of dataset paths must match the number of dataset names."
            )

        self.dataset_paths = dataset_paths
        self.dataset_names = dataset_names
        self.dataset_type = dataset_type
        self.dialogs = []
        self.device = get_device()

        # Load each dataset
        for path, name in zip(dataset_paths, dataset_names):
            logger.info(f"Loading dataset from {path}")
            self.dialogs.extend(load_dataset(path))
            logger.info(f"{name} dataset loaded successfully.")

        self.sentence_model = SentenceTransformer(sentence_model, device=str(self.device))
        logger.info(f"Load pretrained sentiment analysis model: {sentiment_model}.")
        self.sentiment_model = pipeline("sentiment-analysis", model=sentiment_model, device=self.device)

        self.augment_data = augment_data
        self.backtranslation = None

        if self.augment_data and p > 0:
            logger.info(f"Load backtranslation models: {src_translator} - {tgt_translator}.")
            self.backtranslation = BackTranslation(src_translator=src_translator, tgt_translator=tgt_translator, p=p)

        self.graphs = []
        set_verbosity_error()  # disable info and warning logging (progress bar and updates)

    def _process_graph(self, edus, speakers_list, relations):
        graph_data = HeteroData()
        speakers = {}
        speakers_ids = []

        text_embeddings = self.sentence_model.encode(edus, batch_size=BATCH_SIZE, convert_to_tensor=True,
                                                     show_progress_bar=False).to(self.device)
        try:
            sentiments = self.sentiment_model(edus, batch_size=BATCH_SIZE)
        except IndexError:
            sentiments = [{'label': 'NEU'} for _ in range(len(edus))]

        sentiment_labels = [SENTIMENTS[sentiment["label"][:3].upper()] for sentiment in sentiments]
        sentiment_labels = torch.tensor(np.array(sentiment_labels), dtype=torch.int8, device=self.device)

        if str(self.device) != 'cpu':
            getattr(torch, str(self.device)).empty_cache()

        for speaker in speakers_list:
            if speaker not in speakers:
                speakers[speaker] = len(speakers)
            speakers_ids.append(speakers[speaker])

        speakers_ids = torch.tensor(speakers_ids, dtype=torch.int8, device=self.device)
        sentiment_labels = sentiment_labels.unsqueeze(1).repeat(1, text_embeddings.shape[-1])
        speakers_ids = speakers_ids.unsqueeze(1).repeat(1, text_embeddings.shape[-1])

        node_features = torch.cat([text_embeddings, sentiment_labels, speakers_ids], dim=-1)
        graph_data["edu"].x = node_features

        if str(self.device) != 'cpu':
            getattr(torch, str(self.device)).empty_cache()

        edge_dict = defaultdict(list)

        for relation in relations:
            edge_dict[relation["type"]].append((relation["x"], relation["y"]))

        for rel_type, edges in edge_dict.items():
            graph_data["edu", rel_type, "edu"].edge_index = torch.tensor(edges, dtype=torch.long,
                                                                         device=self.device).t().contiguous()

        self.graphs.append(graph_data)

    def __call__(self):
        """
        Build a heterogeneous graph from the dataset where edges represent relationships between EDUs.
        """
        logger.info(
            f"Starting graph construction for dataset/s {self.dataset_names}..."
        )

        idx = 0
        for dialog in tqdm(self.dialogs, desc="Processing dialogs"):
            idx += 1
            if idx < 314:
                continue
            augmented_edus = []
            edus = []
            speakers_list = []
            augmented = False

            for edu in dialog["edus"]:
                edus.append(edu['text'])
                speakers_list.append(edu['speaker'])

                if self.backtranslation is not None:
                    augmented_text = self.backtranslation(edu['text'])
                    augmented_edus.append(augmented_text)
                    augmented = augmented_text != edu['text']

            if augmented:
                self._process_graph(augmented_edus, speakers_list, dialog['relations'])

            self._process_graph(edus, speakers_list, dialog['relations'])

        logger.info("Graph construction completed successfully.")

    def save_graphs(self, path: str, graph: Optional[HeteroData] = None):
        """
        Save the constructed graph to a file.

        Args:
            path (str): Path where the graphs will be saved.
            graph: Specific graph to save. Optional.
        """
        if len(self.graphs) == 0 and graph is None:
            logger.warning(
                "No graph to save. Please run the graph construction first."
            )
            return

        path = os.path.join(path, self.dataset_type)
        logger.info(f"Saving graphs to {path}")

        if graph is None:
            os.makedirs(path, exist_ok=True)

            for idx, graph in enumerate(self.graphs):
                graph_path = os.path.join(path, f"{str(idx)}.pt")
                torch.save(graph, graph_path)
        else:
            torch.save(graph, path)
        logger.info("Graphs saved successfully.")

    def display_graph(self, idx: int):
        """
        Display the graph using networkx and matplotlib. If display is not possible,
        save the graph to a file and log the path.
        """
        if len(self.graphs) == 0:
            logger.warning(
                "No graph to display. Please run the graph construction first."
            )
            return

        return display_graph(self.graphs[idx], "-".join(self.dataset_names))


if __name__ == "__main__":
    builder = GraphBuilder(dataset_paths='../data/STAC/test_subindex.json', dataset_names='AAA', dataset_type='test')
    builder()
