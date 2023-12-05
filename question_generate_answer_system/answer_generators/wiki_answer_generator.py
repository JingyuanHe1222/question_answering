import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import re
import qa_utils
from answer_generators.answer_generator import AnswerGeneratorWithBackup


class WikiAnswerGenerator(AnswerGeneratorWithBackup):
    def __init__(
        self,
        article_filename,
        model_name="deepset/roberta-base-squad2",
    ):
        super().__init__(
            article_filename, model_name, AutoTokenizer, AutoModelForQuestionAnswering
        )
        self._register_backup_model(
            backup_model_name="deepset/bert-base-cased-squad2",
            backup_tokenizer_source=BertTokenizer,
            backup_model_source=BertForQuestionAnswering,
        )

    def generate_answer(self, question):
        question = self._preprocess_question(question)
        if qa_utils.is_boolean_question(question):
            print("encountered yes/no question")
            # return "Yes"

        def callback(model, tokenizer, question):
            return WikiAnswerGenerator._answer_from_article(
                article=self._load_article(),
                question=question,
                tokenizer=tokenizer,
                model=model,
                device=self.device,
            )

        return self._generate_questions_concurrent(question=question, callback=callback)

    def _answer_from_article(article, question, tokenizer, model, device):
        # paragraphs = qa_utils.split_into_paragraphs(article, tokenizer)
        paragraphs = qa_utils.split_article_to_sentences_nltk(article)
        # Iterate through each paragraph to find the best answer
        max_score = -float("inf")
        best_answer = ""
        for paragraph in paragraphs:
            inputs = tokenizer.encode_plus(question, paragraph, return_tensors="pt")
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            outputs = model(**inputs)
            start_score = torch.max(outputs.start_logits)
            end_score = torch.max(outputs.end_logits)

            # Aggregate start and end scores
            score = start_score + end_score
            if score > max_score:
                max_score = score
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits)
                input_ids = inputs["input_ids"].tolist()[0]
                best_answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(
                        input_ids[answer_start : answer_end + 1]
                    )
                )

        return best_answer

    def _preprocess_question(self, question):
        question = question.strip()
        # Remove leading and trailing characters that are not letters
        question = re.sub(r"^[^a-zA-Z]*|[^a-zA-Z]*$", "", question)
        # Append a question mark
        question += "?"
        return question
