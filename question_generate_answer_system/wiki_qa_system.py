import torch
import os
import concurrent.futures
from pathlib import Path
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import wikipediaapi


class WikiQASystem:
    def __init__(
        self,
        article_filename,
        model_name="deepset/bert-base-cased-squad2",
        model_cache_dir="model_cache",
        use_backup_model=False,
    ):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="11411_nlp_team",
        )
        self.model_cache_dir = model_cache_dir
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

        self.article = self._download_article_from_wikipedia(article_filename)
        # print(self.article)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        self.tokenizer, self.model = self._build_tokenizer_and_model(
            model_name=model_name,
            tokenizer_source=BertTokenizer,
            model_source=BertForQuestionAnswering,
        )
        self.use_backup_model = use_backup_model
        if use_backup_model:
            self._init_backup()

    def _build_tokenizer_and_model(self, model_name, tokenizer_source, model_source):
        tokenizer_path = os.path.join(self.model_cache_dir, model_name)
        model_path = os.path.join(self.model_cache_dir, model_name)
        tokenizer = tokenizer_source.from_pretrained(
            tokenizer_path if os.path.exists(tokenizer_path) else model_name,
            cache_dir=self.model_cache_dir,
        )
        model = model_source.from_pretrained(
            model_path if os.path.exists(model_path) else model_name,
            cache_dir=self.model_cache_dir,
        ).to(self.device)
        return tokenizer, model

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
        self.backup_tokenizer, self.backup_model = self._build_tokenizer_and_model(
            model_name=backup_model_name,
            tokenizer_source=backup_tokenizer_source,
            model_source=backup_model_source,
        )

    def _download_article_from_wikipedia(self, article_name):
        return self.wiki_wiki.page(article_name).text

    def _split_into_paragraphs(self, text, max_length=400):
        """
        Splits the text into paragraphs and returns a list of paragraphs.
        Each paragraph's token length is limited to max_length.
        """
        paragraphs = [p for p in text.split("\n") if p]
        short_paragraphs = []
        for paragraph in paragraphs:
            tokens = self.tokenizer.tokenize(paragraph)
            start_token = 0
            while start_token < len(tokens):
                end_token = min(start_token + max_length, len(tokens))
                short_paragraph = tokens[start_token:end_token]
                short_paragraphs.append(
                    self.tokenizer.convert_tokens_to_string(short_paragraph)
                )
                start_token = end_token
        return short_paragraphs

    def generate_answer(self, question):
        question = self._preprocess_question(question)
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
        paragraphs = self._split_into_paragraphs(self.article)
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
        return question.strip()
