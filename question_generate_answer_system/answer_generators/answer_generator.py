import concurrent.futures
from qa_generator import QAGeneratorWithCache
import qa_utils


class AnswerGeneratorWithBackup(QAGeneratorWithCache):
    def __init__(
        self,
        article_name,
        model_name,
        tokenizer_source,
        model_source,
        model_cache_dir=".model-cache",
    ):
        super().__init__(article_name, model_name, model_cache_dir)
        self.primary_tokenizer, self.primary_model = self._load_tokenizer_and_model(
            model_name=model_name,
            tokenizer_source=tokenizer_source,
            model_source=model_source,
        )
        self.backup_models = []

    def _register_backup_model(
        self, backup_model_name, backup_tokenizer_source, backup_model_source
    ):
        backup_tokenizer, backup_model = self._load_tokenizer_and_model(
            model_name=backup_model_name,
            tokenizer_source=backup_tokenizer_source,
            model_source=backup_model_source,
        )
        self.backup_models.append((backup_tokenizer, backup_model))

    def _generate_questions_concurrent(self, question, callback):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.backup_models) + 1
        ) as executor:
            future_primary = executor.submit(
                callback,
                self.primary_model,
                self.primary_tokenizer,
                question,
            )

            # Submit tasks for each backup model
            backup_futures = [
                executor.submit(callback, backup_model, backup_tokenizer, question)
                for backup_tokenizer, backup_model in self.backup_models
            ]

            # Check primary model result first
            primary_answer = future_primary.result()
            if primary_answer not in ["[CLS]", "<s>"]:
                return primary_answer

            # Iterate through backup model results
            for future in backup_futures:
                backup_answer = future.result()
                if backup_answer not in ["[CLS]", "<s>"]:
                    return backup_answer

            ir_backup_output = qa_utils.extract_context_information_retrival(
                self._load_article(), question, 1
            )
            if ir_backup_output is not None or len(ir_backup_output) > 0:
                return ir_backup_output
            return "NO_ANSWER_GENERATED"
