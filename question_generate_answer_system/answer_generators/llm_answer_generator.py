from transformers import AutoModelForCausalLM, AutoTokenizer
import qa_utils
from answer_generators.answer_generator import QAGeneratorWithCache


class LLMAnswerGenerator(QAGeneratorWithCache):
    def __init__(
        self,
        article_filename,
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        beam_width=3,
    ):
        super().__init__(article_filename, model_name)
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name, AutoTokenizer, AutoModelForCausalLM
        )
        self.article = self._load_article()
        self.beam_width = beam_width

    def generate_answer(self, question):
        print("generating answer from LLM")
        context = qa_utils.extract_context_information_retrival(
            self.article, question, self.beam_width
        )

        # Combine the context and the question into a single string
        input_text = (
            f"Context: {context} Question: {question} Answer with 'Yes' or 'No'."
        )

        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        # Generate an answer from the model
        output = self.model.generate(input_ids, max_length=512, num_return_sequences=1)

        # Decode and return the model's output
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if output.lower().startswith("yes") or output.lower().startswith("no"):
            return output
        parts = output.split("Answer:")

        if len(parts) > 1:
            return parts[1].strip()
        return output
