# src/trainer/trainer.py
from transformers import Trainer, TrainingArguments
from typing import Dict

class LlamaTrainer:
    """
    Model eğitimini yöneten sınıf
    """
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        eval_dataset=None
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainer = None

    def create_trainer(self):
        """Trainer nesnesini oluşturur"""
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

    def train(self):
        """Modeli eğitir"""
        if self.trainer is None:
            self.create_trainer()
        return self.trainer.train()

    def save_model(self, output_dir: str):
        """Modeli kaydeder"""
        self.trainer.save_model(output_dir)