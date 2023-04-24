import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from .base_search_algorithm import BaseSearchAlgorithm
from knowledge_base_search import logger

class AutoSearch(BaseSearchAlgorithm):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled_output = self._mean_pooling(outputs, inputs['attention_mask'])
        pooled_output = F.normalize(pooled_output, p=2, dim=1).numpy()
        return pooled_output
