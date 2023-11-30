from qa_generator import QAGeneratorWithCache
import qa_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class T5QuestionGenerator(QAGeneratorWithCache):
    def __init__(
        self,
        article_filename,
        model_name="potsawee/t5-large-generation-squad-QuestionAnswer",
        local_model_path="./.model-cache/latest_model.pt",
        questions_to_generate=10,
    ):
        super().__init__(article_filename, model_name)
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name=model_name,
            tokenizer_source=AutoTokenizer,
            model_source=AutoModelForSeq2SeqLM,
        )
        if local_model_path:
            checkpoint = torch.load(
                local_model_path, map_location=torch.device(self.model.device)
            )
            self.model.load_state_dict(checkpoint["weights"])
            print("successfully loaded checkpoint")
        self.questions_to_generate = questions_to_generate
        self.phrase_to_remove = "generate question"

    def generate_questions(self):
        article = self._load_article()
        sentences = qa_utils.split_article_to_sentences_nltk(article)
        questions = []
        for sentence in sentences:
            input_text = sentence
            inputs = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            )
            outputs = self.model.generate(
                inputs, max_length=64, num_beams=4, early_stopping=True
            )
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = self.process_question(question)
            if question:
                questions.append(question)

            if len(questions) >= self.questions_to_generate:
                break

        return questions

    def generate_questions(self):
        article = self._load_article()
        paragraphs = qa_utils.split_into_paragraphs(article, self.tokenizer)
        questions = []
        for paragraph in paragraphs:
            input_text = "generate question: " + paragraph
            inputs = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=512, truncation=True
            )
            outputs = self.model.generate(
                inputs, max_length=64, num_beams=4, early_stopping=True
            )
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = self.process_question(question)
            if question:
                questions.append(question)

            if len(questions) >= self.questions_to_generate:
                break

        return questions

    def process_question(self, question):
        cleaned_question = question.replace(self.phrase_to_remove, "").strip()
        cleaned_question = " ".join(cleaned_question.split())

        parts = cleaned_question.split("?")
        if len(parts) > 1:
            return parts[0].strip() + "?"
        else:
            return None
