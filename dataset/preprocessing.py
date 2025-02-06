import json

import torch
from transformers import AutoModel, AutoTokenizer, pipeline


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# --- NLP PROCESSING ---
def process_text(edu):
    return edu.strip()


def get_sentiment(edu):
    sentiment_pipeline = pipeline("sentiment-analysis")
    res = sentiment_pipeline(edu)[0]
    if res["label"] == "POSITIVE":
        score = res["score"]
    else:
        score = -res["score"]
    return score


# -- EMBEDDING --
class SequenceBERT(torch.nn.Module):
    def __int__(self, model_name="answerdotai/ModernBERT-base"):
        super().__int__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, edus):
        tokens = self.tokenizer(edus, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.bert(**tokens).last_hidden_state.mean(dim=1)
        return embedding


sequence_bert = SequenceBERT()


# -- FEATURES EXTRACTION --
def extract_features(edu_list):
    embeddings = []
    sentiments = []
    speakers = []

    for edu in edu_list:
        text = process_text(edu["text"])
        speaker = edu["speaker"]

        embedding = sequence_bert.forward([text]).numpy()
        sentiment = get_sentiment(text)

        embeddings.append(embedding)
        sentiments.append(sentiment)
        speakers.append(speaker)

    return embeddings, sentiments, speakers
