from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from src.config.model_config import ModelArguments
import torch

class LlamaPeftModel:
    """
    PEFT modelini oluşturan ve yöneten sınıf
    """
    def __init__(self, model_args: ModelArguments):
        self.model_args = model_args
        self.tokenizer = None
        self.model = None
        self.peft_config = None

    def load_model(self):
        """Model ve tokenizer'ı yükler"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )

        # Burada 'pad_token' ekliyoruz:
        if self.tokenizer.pad_token_id is None:
            # Yöntem 1: Yeni bir [PAD] token ekleyerek
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            # Yöntem 2: İsterseniz, eos_token'ı pad_token olarak kullanabilirsiniz:
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Eğer yeni bir özel token eklediyseniz, modelin embedding boyutunu güncellemeniz gerekir:
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_peft_config(self):
        """PEFT konfigürasyonunu hazırlar"""
        if self.model_args.use_lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_args.lora_rank,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                target_modules=self.model_args.target_modules
            )

    def get_peft_model(self):
        """PEFT modelini oluşturur ve döndürür"""
        if self.model is None:
            self.load_model()
        if self.peft_config is None:
            self.prepare_peft_config()
        
        return get_peft_model(self.model, self.peft_config)