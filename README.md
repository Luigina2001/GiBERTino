<div align="center">
    <img src="https://github.com/user-attachments/assets/52b8223d-5740-47c4-b699-9497c6c52a9b" alt="GiBERTino logo" width="300"/>
    <h1>GiBERTino</h1>
    <h3>Supervised extraction of sub-dialogues from multi-part dialogues</h3>
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

We confront our method with different state-of-the-arts methodologies on link and relation prediction, analyzing their performance both intra and cross domain. 

Here is a comparison between GiBERTino and the unsupervised approach from [Cimino et al., 2024](https://aclanthology.org/2024.sigdial-1.26/) on link prediction.

| **Model**                        | **F₁ (STAC)** | **P (STAC)** | **R (STAC)** | **F₁ (MOLWENI-clean)** | **P (MOLWENI-clean)** | **R (MOLWENI-clean)** |
|:--------------------------------:|:-------------:|:------------:|:------------:|:-----------------------:|:----------------------:|:----------------------:|
| DS-DP + STL + P₍dist(d)₎         |     0.573     |    0.551     |    0.597     |          0.747          |         0.701          |         0.799          |
| DS-FLOW + STL + P₍dist(d)₎       |     0.581     |    0.571     |    0.592     |          0.729          |         0.691          |         0.772          |
| DS-DP + STL + P₍dist(d)₎         |     0.567     |    0.534     |    0.604     |          0.729          |         0.691          |         0.772          |
| IMP-GS-STAC                      |   **0.819**   |    0.704     |  **0.978**   |          0.812          |         0.695          |       **0.977**        |
| IMP-GS-MOL                       |     0.814     |  **0.710**   |    0.952     |        **0.815**        |       **0.719**        |         0.941          |

Here follows a comparison between GiBERTino and the methdology from [Chi et al., 2023](https://arxiv.org/abs/2306.15103) on both link prediction and relation classification. STAC/MOL means that the training set was STAC and the evaluation set was MOLWENI. 
<div align="center"> 
    
| **Model**            | **Link F₁** | **Rel. F₁** | **Link F₁** | **Rel. F₁** | **Link F₁** | **Rel. F₁** | **Link F₁** | **Rel. F₁** |
|:--------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|                      | **STAC/STAC** |            | **MOL/MOL** |            | **STAC/MOL** |            | **MOL/STAC** |            |
| Chi et al., 2023 |    0.744    |    0.596    |  **0.835**  |    0.599    |    0.645    |    0.380    |    0.506    |    0.316    |
| **GiBERTino**        |  **0.819**  |    0.041    |    0.815    |    0.032    |  **0.812**  |    0.048    |  **0.814**  |    0.019    |

</div>

Here follows a comparison between GiBERTino and [LLamipa3p+p](https://arxiv.org/abs/2406.18256) on both link prediction and relation classification.. STAC/MOL means that the training set was STAC and the evaluation set was MOLWENI. 
<div align="center"> 
    
| **Model**         | **Link F₁** | **Rel. F₁** | **Link F₁** | **Rel. F₁** |
|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|
|                   | **STAC/STAC** |             | **STAC/MOL** |             |
| LLamipa3+p        |    0.775    |    0.607    |    0.712    |    0.405    |
| **GiBERTino**     |  **0.819**  |    0.041    |  **0.812**  |    0.048    |

</div>

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
