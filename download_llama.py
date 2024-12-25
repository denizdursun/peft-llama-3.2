from transformers import AutoModelForCausalLM, AutoTokenizer

# Model adını belirtiyoruz
model_name = "meta-llama/LLaMA-3.2-1B"  # Örnek model adı

# Tokenizer ve modelin indirilmesi
print("Tokenizer indiriliyor...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model indiriliyor...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model ve tokenizer başarıyla indirildi!")