from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_search_algorithm import BaseSearchAlgorithm
from knowledge_base_search import logger


class TfidfSearch(BaseSearchAlgorithm):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.indexed_documents = None

    def index_documents(self, documents):
        self.indexed_documents = self.vectorizer.fit_transform(documents)
        logger.info("Indexed documents using TfidfVectorizer.")

    def search(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(self.indexed_documents, query_vector)
        return similarity_scores
