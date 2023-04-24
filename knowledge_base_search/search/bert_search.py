import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_search_algorithm import BaseSearchAlgorithm
from knowledge_base_search import logger

class BERTSearch(BaseSearchAlgorithm):
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.indexed_documents = []

    def index_documents(self, documents):
        self.indexed_documents = self._embed_documents(documents)
        logger.info("Indexed documents using BERT.")

    def search(self, query):
        query_embedding = self._embed_text(query).reshape(1, -1)
        indexed_documents_flat = self.indexed_documents.reshape(self.indexed_documents.shape[0], -1)
        similarity_scores = cosine_similarity(indexed_documents_flat, query_embedding)
        return similarity_scores


    def _embed_documents(self, documents):
        embeddings = []
        for document in documents:
            embedding = self._embed_text(document)
            embeddings.append(embedding)
        return np.stack(embeddings)

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state.mean(axis=1).detach().numpy()
        return pooled_output
