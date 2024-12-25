# modeli dağıtmak için
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class InferenceModel:
    def __init__(self, model_path: str):
        self.config = PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16  # Hafıza kullanımını azaltmak için
        )
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path
        )
        
    def generate(self, prompt: str, max_length: int = 128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self, output_dir: str):
        """Modeli ve tokenizer'ı kaydet"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)