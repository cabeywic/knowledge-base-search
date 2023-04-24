import openai
from knowledge_base_search import logger


class AnswerGenerator:
    def __init__(self, openai_api_key, model_name="text-davinci-003", max_tokens=100, temperature=0.5):
        openai.api_key = openai_api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_answer(self, query, context, warning_threshold=0.5):
        if context["similarity_score"] < warning_threshold:
            logger.warning(f"Warning: The document's similarity score ({context['similarity_score']}) is below the threshold ({warning_threshold}). The answer might not be accurate.")
        
        prompt = f"{query}\n\nContext: {context['content']}\n\nAnswer:"
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].text.strip()
