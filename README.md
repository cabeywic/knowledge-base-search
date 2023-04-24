# Knowledge Base Search

This project provides an efficient and scalable solution to search and query a large knowledge base of documents. It allows users to search for information easily by leveraging advanced NLP techniques like BERT embeddings.

## Features

- Organized code structure following SOLID principles
- BERT search for semantic similarity between queries and documents
- Preprocessing using SpaCy for efficient text processing
- Caching system to store preprocessed data and search algorithm instances for faster subsequent searches
- Logging to track search-related information and potential issues

## Methodology

The Knowledge Base Search tool employs a two-step process to find relevant documents and generate human-readable answers:

1. **Semantic Search**: The tool preprocesses and indexes the input documents using advanced NLP techniques like BERT/MiniLM embeddings or a custom search implementation. These embeddings capture the semantic meaning of the text, allowing the search algorithm to find documents that are not just textually similar, but also semantically related to the input query. This approach ensures a more accurate and context-aware selection of relevant documents.

2. **Answer Generation**: After retrieving the most relevant documents, the tool integrates with OpenAI's Chat GPT API to generate human-readable answers based on the provided context. By only sending the relevant context, we can reduce the cost and improve the performance of the API calls, while ensuring that the generated answers are accurate and contextually appropriate.

This methodology is designed to be easily extensible and customizable, allowing users to implement their own search algorithms or NLP models to tailor the solution to their specific use case.


## Installation

To set up the project, follow these steps:

1. Clone the repository:

```sh
git clone https://github.com/your_username/knowledge_base_search.git
```

2. Change the directory:
```sh
cd knowledge_base_search
```

3. Create a virtual environment:
- For Windows:
```sh
python -m venv venv
```
- For Linux/Mac:
```sh
python3 -m venv venv
```

4. Activate the virtual environment:
- For Windows:
```sh
venv\Scripts\activate
```
- For Linux/Mac:
```sh
source venv/bin/activate
```

5. Install the required packages:
```sh
pip install -r requirements.txt
```

6. Create a `.env` file in the root of your project and add the openai_api_key variable. Replace `<your_api_key>` with your actual API key:

```sh
openai_api_key=<your_api_key>
```

## Usage

1. Add your documents in JSON format to the `data/raw_data/documents.json` file.

2. Update the `main.py` file with your query and other necessary modifications.

3. Run the `main.py` script:

```sh
python main.py
```

This will load the documents, preprocess them, and index them using the specified search algorithm (e.g., BERT). Then, it will search for relevant documents based on your query and return the top matching results.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to improve the project.

