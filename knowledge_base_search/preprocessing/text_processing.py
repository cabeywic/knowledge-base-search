import spacy
from gensim.models.phrases import Phrases, Phraser

class TextProcessor:
    def __init__(self, documents=None):
        # python -m spacy download en_core_web_sm
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        if documents:
            self.phrase_model = self._build_phrase_model(documents)
        else:
            self.phrase_model = None

    def process_text(self, text):
        tokens = self._tokenize_and_lemmatize(text)
        if self.phrase_model:
            tokens = self.phrase_model[tokens]
        return tokens

    def _build_phrase_model(self, documents):
        tokenized_documents = [self._tokenize_and_lemmatize(doc) for doc in documents]
        phrases = Phrases(tokenized_documents, min_count=5, threshold=10)
        return Phraser(phrases)

    def _tokenize_and_lemmatize(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        return tokens
