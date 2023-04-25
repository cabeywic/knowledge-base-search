from knowledge_base_search.knowledge_base import KnowledgeBase
from knowledge_base_search.answer_generator import AnswerGenerator
from knowledge_base_search.search.minilm_search import MiniLMSearch
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Example search query
query = "Who is the CEO and the Managing Director?"

search_algorithm = MiniLMSearch()
# Load documents
knowledge_base = KnowledgeBase(search_algorithm, "data/raw_data/documents.json", cache=True)

# Search using BERTSearch
top_n = 2
top_documents = knowledge_base.search(query, top_n)
top_document = top_documents[0]
print(f"Top Document[{type(search_algorithm).__name__}]: ", top_document['content'])

# Set your OpenAI API key
openai_api_key = os.environ.get("openai_api_key")

# Create an AnswerGenerator instance
answer_generator = AnswerGenerator(openai_api_key)

# Generate an answer using the most relevant document found by BERTSearch
answer = answer_generator.generate_answer(query, top_document)
print("\n\nGenerated answer:", answer)
