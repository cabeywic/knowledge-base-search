# Knowledge Base Search

This project provides an efficient and scalable solution to search and query a large knowledge base of documents. It allows users to search for information easily by leveraging advanced NLP techniques like BERT embeddings.

## Features

- Organized code structure following SOLID principles
- BERT search for semantic similarity between queries and documents
- Preprocessing using SpaCy for efficient text processing
- Caching system to store preprocessed data and search algorithm instances for faster subsequent searches
- Logging to track search-related information and potential issues

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

