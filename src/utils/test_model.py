from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model(model_path: str):
    """Eğitilmiş modeli yükler"""
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    return model, tokenizer

def generate_text(prompt: str, model, tokenizer, max_length: int = 128):
    """Verilen prompt için metin üretir"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test örneği
if __name__ == "__main__":
    model_path = "./results/checkpoint-best"  # En iyi checkpoint'i kullan
    model, tokenizer = load_model(model_path)
    
    test_prompts = [
        "Yapay zeka nedir?",
        "Python programlama dilinin özellikleri nelerdir?",
        # Daha fazla test promptu ekleyin
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_text(prompt, model, tokenizer)
        print(f"Response: {response}")