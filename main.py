import logging
import argparse
import torch

#from transformers import TrainingArguments  # transformers'dan TrainingArguments import ediyoruz

from src.config.model_config import ModelArguments, TrainingArguments
from src.data.data_module import LlamaDataset
from src.model.peft_model import LlamaPeftModel
from src.trainer.trainer import LlamaTrainer

logging.basicConfig(level=logging.DEBUG)  # Konsola log basmayı aktif eder
logger = logging.getLogger(__name__)

def parse_args():
    """
    Komut satırından veya manuel olarak parametreleri alma işlemini gerçekleştirir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Eğitim sonrasında kaydedilecek klasör")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Yerel dosyadan eğitime giriş yapmak için json dosyası")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Hugging Face dataset ismi (örn: 'wikitext' vb.)")
    parser.add_argument("--use_local_data", action="store_true",
                        help="Yerel veri dosyası mı kullanılacak?")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Eğitim epoch sayısı")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Öğrenme oranı")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Kaç adımda bir loglama yapılacak?")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1) Model argümanları (Yerel model yolunu buraya ekliyoruz)
    model_args = ModelArguments(
        model_name_or_path=r"C:\Users\Administrator\.cache\huggingface\hub\models--meta-llama--LLaMA-3.2-1B\snapshots\4e20de362430cd3b72f300e6b0f18e50e7166e08"  # Yerel model klasör yolu
    )

    # 2) Eğitim argümanları
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        logging_dir=f"{args.output_dir}/logs",
        report_to=["tensorboard"],
        fp16=False  # GPU kullanılmadığı için False
    )

    logger.info("Eğitim Argümanları Yüklendi: %s", training_args)

    # 3) Modeli yükle (Yerel yol üzerinden)
    llama_peft_model = LlamaPeftModel(model_args=model_args)
    llama_peft_model.load_model()
    llama_peft_model.prepare_peft_config()
    peft_model = llama_peft_model.get_peft_model()

    # 4) Dataset oluştur (Yerel dosyadan veya Hugging Face dataset)
    if args.use_local_data and args.train_file is not None:
        train_dataset = LlamaDataset.from_local_file(
            file_path=args.train_file,
            tokenizer=llama_peft_model.tokenizer,
            split="train",
            max_length=512
        )
        eval_dataset = None
    elif args.dataset_name:
        train_dataset = LlamaDataset.from_huggingface(
            dataset_name=args.dataset_name,
            tokenizer=llama_peft_model.tokenizer,
            split="train",
            max_length=512
        )
        eval_dataset = None
    else:
        raise ValueError("Eğitim verisi belirtilmedi. --use_local_data veya --dataset_name parametresini ayarlayın.")

    logger.info(f"Veri Yükleme Tamamlandı. Eğitim örnek sayısı: {len(train_dataset)}")

    # 5) Trainer oluştur ve eğitimi başlat
    llama_trainer = LlamaTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None
    )

    logger.info("Eğitim Başlıyor...")
    train_result = llama_trainer.train()
    logger.info("Eğitim Tamamlandı.")

    # 6) Modeli kaydet
    llama_trainer.save_model(args.output_dir)
    logger.info(f"Model '{args.output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
