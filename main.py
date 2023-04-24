from knowledge_base_search.knowledge_base import KnowledgeBase
from knowledge_base_search.answer_generator import AnswerGenerator
from knowledge_base_search.search.minilm_search import MiniLMSearch
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Example search query
query = "Who is the CEO and the Managing Director?"

# Load documents
knowledge_base_bert = KnowledgeBase(MiniLMSearch(), "data/raw_data/documents.json", cache=True)

# Search using BERTSearch
top_n = 2
top_documents_bert = knowledge_base_bert.search(query, top_n)
print(f"Top {top_n} documents for the query using BERTSearch:", top_documents_bert)

# Set your OpenAI API key
openai_api_key = os.environ.get("openai_api_key")

# Create an AnswerGenerator instance
answer_generator = AnswerGenerator(openai_api_key)

# Generate an answer using the most relevant document found by BERTSearch
top_document = top_documents_bert[0]
answer = answer_generator.generate_answer(query, top_document)
print("\n\n\nGenerated answer:", answer)
