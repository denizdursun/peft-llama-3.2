import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Eğitilmiş modelin kayıt yolu (örn: './results/final_model')
        """
        logger.info("Model yükleniyor...")
        self.config = PeftConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path
        )
        logger.info("Model yüklendi ve kullanıma hazır!")

    def chat(self, prompt: str, max_length: int = 512, temperature: float = 0.7):
        """
        Verilen prompt ile sohbet et
        
        Args:
            prompt: Kullanıcı mesajı
            max_length: Maksimum yanıt uzunluğu
            temperature: Yanıt çeşitliliği (0.0-1.0)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # Eğitilmiş model yolunu belirtin
    model_path = "./results/final_model"
    
    # ChatBot'u başlat
    chatbot = ChatBot(model_path)
    
    print("ChatBot hazır! Çıkmak için 'quit' yazın.")
    
    while True:
        user_input = input("\nSiz: ")
        if user_input.lower() == 'quit':
            break
            
        try:
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()