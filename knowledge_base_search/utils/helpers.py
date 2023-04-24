import json
from knowledge_base_search.preprocessing.document import Document

def load_documents(file_path):
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    documents = [Document(doc["id"], doc["content"]) for doc in raw_data]
    return documents
