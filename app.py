import streamlit as st
from knowledge_base_search.knowledge_base import KnowledgeBase
from knowledge_base_search.search.minilm_search import MiniLMSearch
from knowledge_base_search.answer_generator import AnswerGenerator


st.set_page_config(page_title="Knowledge Base Search", layout="wide")

@st.cache(allow_output_mutation=True)
def load_knowledge_base():
    return KnowledgeBase(MiniLMSearch(), "data/raw_data/documents.json", cache=True)

knowledge_base_bert = load_knowledge_base()
openai_api_key = st.secrets["openai_api_key"]
answer_generator = AnswerGenerator(openai_api_key)

@st.cache(show_spinner=False)
def search_knowledge_base(query, top_n):
    top_documents = knowledge_base_bert.search(query, top_n=top_n)
    generated_answer = ""
    if top_documents:
        most_relevant_document = top_documents[0]
        print(most_relevant_document)
        generated_answer = answer_generator.generate_answer(query, most_relevant_document)
    return generated_answer, top_documents

st.title("🔎📒 Knowledge Base Search")

# Input field for user query
query = st.text_input("Enter your query:")

# Number input for the top N results
top_n = st.number_input("Number of top results:", min_value=1, max_value=10, value=5)

# Search button
if st.button("Search"):
    with st.spinner("Searching..."):
        generated_answer, results = search_knowledge_base(query, top_n)
        
        if generated_answer:
            st.header("Generated Answer:")
            st.write(generated_answer)
        
        if results:
            # Expander for search results
            with st.expander("Search Results"):
                for result in results:
                    st.subheader(f"{result['id']} - Confidence : {result['similarity_score'][0]:.2f}")
                    st.write(result['content'])
                    st.write("---")
        else:
            st.warning("No relevant documents found.")

