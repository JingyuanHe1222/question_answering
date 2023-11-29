import torch
import concurrent.futures
from qa_generator import QAGeneratorWithCache
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import re
import qa_utils


class WikiAnswerGenerator(QAGeneratorWithCache):
    def __init__(
        self,
        article_filename,
        model_name="deepset/bert-base-cased-squad2",
        use_backup_model=False,
    ):
        super().__init__(article_filename, model_name)
        self.tokenizer, self.model = self._load_tokenizer_and_model(
            model_name=model_name,
            tokenizer_source=BertTokenizer,
            model_source=BertForQuestionAnswering,
        )
        self.yes_no_pattern = re.compile(
            r"^(is|are|can|do|does|did|will|would|should|has|have|had|am|were|was)\b"
        )
        self.use_backup_model = use_backup_model
        if use_backup_model:
            self._init_backup()

    def _init_backup(
        self,
        backup_model_name="deepset/roberta-base-squad2",
        backup_tokenizer_source=AutoTokenizer,
        backup_model_source=AutoModelForQuestionAnswering,
    ):
        if not self.use_backup_model:
            self.backup_tokenizer = None
            self.backup_model = None
            return
        self.backup_tokenizer, self.backup_model = self._load_tokenizer_and_model(
            model_name=backup_model_name,
            tokenizer_source=backup_tokenizer_source,
            model_source=backup_model_source,
        )

    def generate_answer(self, question):
        question = self._preprocess_question(question)
        if self._is_boolean_question(question):
            print("encountered yes/no question")
            # return "Yes"
        if self.use_backup_model:
            return self._answer_from_article_parallel(question)
        else:
            return self._answer_from_article(
                question=question, tokenizer=self.tokenizer, model=self.model
            )

    def _answer_from_article_parallel(self, question):
        if not self.use_backup_model:
            return self._answer_from_article(
                question=question, tokenizer=self.tokenizer, model=self.model
            )

        def get_answer(model, tokenizer, question):
            return self._answer_from_article(
                question=question, tokenizer=tokenizer, model=model
            )

        # primary: BERT
        # backup: Roberta
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # run both the primary and backup
            future_primary = executor.submit(
                get_answer,
                self.model,
                self.tokenizer,
                question,
            )
            future_backup = executor.submit(
                get_answer,
                self.backup_model,
                self.backup_tokenizer,
                question,
            )
            primary_answer = future_primary.result()
            if primary_answer == "[CLS]":
                print(
                    "Primary model failed to generate an answer. Trying backup model..."
                )
                return future_backup.result()
            else:
                return primary_answer

    def _answer_from_article(self, question, tokenizer, model):
        paragraphs = qa_utils.split_into_paragraphs(self._load_article(), tokenizer)
        # Iterate through each paragraph to find the best answer
        max_score = -float("inf")
        best_answer = ""
        for paragraph in paragraphs:
            inputs = self.tokenizer.encode_plus(
                question, paragraph, return_tensors="pt"
            )
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            outputs = self.model(**inputs)
            start_score = torch.max(outputs.start_logits)
            end_score = torch.max(outputs.end_logits)

            # Aggregate start and end scores
            score = start_score + end_score
            if score > max_score:
                max_score = score
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits)
                input_ids = inputs["input_ids"].tolist()[0]
                best_answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
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

    def _is_boolean_question(self, question):
        return bool(self.yes_no_pattern.match(question.strip().lower()))
