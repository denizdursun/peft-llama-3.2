from dataclasses import dataclass
from typing import Optional, List
from transformers import TrainingArguments as HfTrainingArguments

@dataclass
class ModelArguments:
    """
    Model için gerekli argümanları tutan sınıf
    """
    model_name_or_path: str
    peft_config_path: Optional[str] = None
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None

class TrainingArguments(HfTrainingArguments):
    """
    Hugging Face'in TrainingArguments sınıfını miras alan özelleştirilmiş sınıf
    """
    def __init__(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        logging_steps: int = 1,
        save_steps: int = 100,
        save_total_limit: int = 3,
        save_strategy: str = "steps",
        evaluation_strategy: str = "no",
        load_best_model_at_end: bool = False,
        do_train: bool = True,
        **kwargs
    ):
        super().__init__(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            do_train=do_train,
            **kwargs
        )