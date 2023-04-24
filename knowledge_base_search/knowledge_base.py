from knowledge_base_search.preprocessing.text_processing import TextProcessor
from knowledge_base_search.utils.helpers import load_documents
from knowledge_base_search import logger

class KnowledgeBase:
    def __init__(self, search_algorithm, documents_path):
        self.documents = load_documents(documents_path)
        self.text_processor = TextProcessor(documents=[doc.content for doc in self.documents])

        processed_documents = [' '.join(self.text_processor.process_text(doc.content)) for doc in self.documents]

        self.search_algorithm = search_algorithm
        self.search_algorithm.index_documents(processed_documents)
        logger.info(f"Set search algorithm to {type(search_algorithm).__name__}.")

    def search(self, query, top_n=5):
        processed_query = ' '.join(self.text_processor.process_text(query))
        similarity_scores = self.search_algorithm.search(processed_query)

        top_n_document_indices = similarity_scores.flatten().argsort()[-top_n:][::-1]
        logger.info(f"Searched for query: {query}. Found {len(top_n_document_indices)} relevant documents.")

        top_n_documents = [
            {"id": self.documents[i].id, "content": self.documents[i].content, "similarity_score": similarity_scores[i]}
            for i in top_n_document_indices
        ]

        return top_n_documents

