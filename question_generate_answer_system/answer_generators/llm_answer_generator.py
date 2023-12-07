import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import qa_utils
from answer_generators.answer_generator import QAGeneratorWithCache


class LLMAnswerGenerator(QAGeneratorWithCache):
    def __init__(
        self,
        article_filename,
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        beam_width=3,
    ):
        super().__init__(
            article_filename, model_name, AutoTokenizer, AutoModelForCausalLM
        )
        self.article = self._load_article()
        self.beam_width = beam_width

    def generate_answer(self, question):
        context = qa_utils.extract_context_information_retrival(
            self.article, question, self.beam_width
        )
        # Combine the context and the question into a single string
        input_text = f"context: {context} question: {question}"

        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate an answer from the model
        output = self.model.generate(input_ids, max_length=512, num_return_sequences=1)

        # Decode and return the model's output
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
