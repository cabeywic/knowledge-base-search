from knowledge_base_search.preprocessing.text_processing import TextProcessor
from knowledge_base_search.utils.helpers import load_documents
from knowledge_base_search import logger
import time 
import os
import pickle


class KnowledgeBase:
    def __init__(self, search_algorithm, documents_path, cache=False):
        self.documents = load_documents(documents_path)
        logger.info(f"Loaded {len(self.documents)} documents.")

        search_algorithm_cache_filename = f"{type(search_algorithm).__name__}_search_algorithm.pickle"
        text_processor_cache_filename = "text_processor.pickle"
        print(os.path.exists(search_algorithm_cache_filename), os.path.exists(text_processor_cache_filename))

        if cache and os.path.exists(search_algorithm_cache_filename) and os.path.exists(text_processor_cache_filename):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            with open(search_algorithm_cache_filename, "rb") as f:
                self.search_algorithm = pickle.load(f)

            with open(text_processor_cache_filename, "rb") as f:
                self.text_processor = pickle.load(f)

            logger.info("Loaded search_algorithm and text_processor from cache.")
        else:
            start_time = time.time()
            self.text_processor = TextProcessor(documents=[doc.content for doc in self.documents])
            if cache:
                with open(text_processor_cache_filename, "wb") as f:
                    pickle.dump(self.text_processor, f)
                logger.info("Saved text_processor to cache.")

            processed_documents = [' '.join(self.text_processor.process_text(doc.content)) for doc in self.documents]
            print(f"Time to process documents: {time.time() - start_time} seconds")

            self.search_algorithm = search_algorithm
            self.search_algorithm.index_documents(processed_documents)

            if cache:
                with open(search_algorithm_cache_filename, "wb") as f:
                    pickle.dump(self.search_algorithm, f)
                logger.info("Saved search_algorithm to cache.")

            print(f"Time to index documents: {time.time() - start_time} seconds")
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

