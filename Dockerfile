FROM python:3.9-slim

WORKDIR /app

# Gerekli kütüphaneleri kopyala ve kur
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama kodlarını kopyala
COPY src/ src/
COPY models/ models/

# Çalıştırma komutu
CMD ["python", "src/utils/inference.py"]