from abc import ABC, abstractmethod

class BaseSearchAlgorithm(ABC):
    @abstractmethod
    def index_documents(self, documents):
        pass

    @abstractmethod
    def search(self, query):
        pass
