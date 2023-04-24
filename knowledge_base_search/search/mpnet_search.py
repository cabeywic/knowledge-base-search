from knowledge_base_search.search.auto_search import AutoSearch


class MPNetSearch(AutoSearch):
    def __init__(self):
        super().__init__(model_name="sentence-transformers/all-mpnet-base-v2")
