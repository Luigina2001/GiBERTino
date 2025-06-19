<div align="center">
    <img src="https://github.com/user-attachments/assets/52b8223d-5740-47c4-b699-9497c6c52a9b" alt="GiBERTino logo" width="300"/>
    <h1>GiBERTino</h1>
    <h3>Supervised extracted of sub-dialogues from multi-part dialogues</h3>
</div>

<p align="center">
 <a href="#"><img src="https://img.shields.io/github/contributors/Luigina2001/GiBERTino?style=for-the-badge" alt="Contributors"/></a>
 <img src="https://img.shields.io/github/last-commit/Luigina2001/GiBERTino?style=for-the-badge" alt="last commit">
</p>
<p align="center">
 <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/></a>
 <a href="#"><img src="https://img.shields.io/github/languages/top/Luigina2001/GiBERTino?style=for-the-badge" alt="Languages"/></a>
</p>

# Introduction

Conversations do not follow a linear logic, they dynamically mix: objections, proposals, questions, answers overlap, generating interactive and complex exchanges. The task of automatically identifying parts of a discourse that follow the same semantic thread becomes challenging, especially when dealing with **multi-part** dialogues,since they involve multiple interlocutors, resulting in overlapping reply-chains and more complex information flows that do not depend simply on temporal sequentiality.

**GiBERTino** addresses this complexity by leveraging pre-trained language models like ModernBERT and graph neural networks to model the nuanced structure of dialogue interactions.  


# Methodology

**GiBERTino** combines a pre-trained transformer model with a Graph Neural Network (GNN) to process multi-party dialogues modeled as graphs. The architecture consists of three main components:

1. **Text Encoding**:  
   Each EDU (Elementary Discourse Unit) is tokenized and encoded using a [ModernBERT model](https://huggingface.co/Alibaba-NLP/gte-modernbert-base). A mean pooling over the last hidden states generates a fixed-size vector for each EDU, which is then concatenated with static node features.

2. **Graph Propagation**:  
   These representations are passed through a multi-layer GNN (either GCN or GraphSAGE). Through message passing across $L$ layers, nodes aggregate context from their neighbors, enabling the model to capture both local and global discourse structure.

3. **Prediction Heads**:  
   - **Link Prediction**: A classifier determines whether an edge exists between two nodes based on their final representations, using cosine similarity and other vector operations.  
   - **Relation Classification**: A second classifier predicts the type of discourse relation (from $C$ classes) between node pairs, using trainable relation embeddings.

Both classifiers are implemented as 3-layer MLPs and trained using standard loss functions (binary cross-entropy for link prediction, cross-entropy for relation classification).

![GNN_arch](https://github.com/user-attachments/assets/a1ffad79-8b3a-4314-9e4e-50e6f5d44b61)


# Results

# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.10` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/Luigina2001/GiBERTino.git
```

## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```

# Citation

If you use this project, please consider citing:  
```
@article{costantenazzaro2025:gibertino,
  author    = {Luigina Costante, Angelo Nazzaro},
  title     = {GiBERTino: Supervised extracted of sub-dialogues from multi-part dialogues},
  year      = {2025},
  institution = {University of Salerno}
}
```
