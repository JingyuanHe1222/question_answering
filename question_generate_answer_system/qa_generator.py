import torch
import os
from pathlib import Path


class QAGeneratorWithCache:
    def __init__(self, model_name, model_cache_dir="model_cache"):
        self.model_cache_dir = model_cache_dir
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

    def _load_tokenizer_and_model(self, model_name, tokenizer_source, model_source):
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
