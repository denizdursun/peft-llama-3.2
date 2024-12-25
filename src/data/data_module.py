from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer

class LlamaDataset(Dataset):
    """
    Veri setini hazırlayan sınıf
    """
    def __init__(
        self,
        data_source: Union[str, Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
        is_local_file: bool = False,
    ):
        """
        Args:
            data_source: Veri kaynağı. Hugging Face dataset adı veya yerel dosya yolu olabilir
            tokenizer: Kullanılacak tokenizer
            max_length: Maximum token uzunluğu
            split: Veri seti bölümü (train, validation, test)
            is_local_file: Yerel dosya kullanılıp kullanılmayacağı
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Veri setini yükle
        if is_local_file:
            if isinstance(data_source, str):
                self.dataset = load_dataset(
                    "json",
                    data_files=data_source,
                    split=split
                )
            else:
                self.dataset = load_dataset(
                    "json",
                    data_files=data_source,
                )[split]
        else:
            self.dataset = load_dataset(data_source, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        """Veri setinden bir örnek döndür"""
        item = self.dataset[idx]
        
        # Instruction ve input'u birleştir
        instruction = item["instruction"]
        input_text = item["input"]
        output = item["output"]
        
        # Prompt formatını oluştur
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        # Tokenize işlemi
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Causal LM için etiketler (labels), input_ids ile aynıdır.
        # Model, shift edip kendi içinde loss hesaplar.
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        **kwargs
    ):
        """Hugging Face'den veri seti yüklemek için yardımcı metod"""
        return cls(
            data_source=dataset_name,
            tokenizer=tokenizer,
            split=split,
            is_local_file=False,
            **kwargs
        )

    @classmethod
    def from_local_file(
        cls,
        file_path: Union[str, Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        **kwargs
    ):
        """Yerel dosyadan veri seti yüklemek için yardımcı metod"""
        return cls(
            data_source=file_path,
            tokenizer=tokenizer,
            split=split,
            is_local_file=True,
            **kwargs
        )