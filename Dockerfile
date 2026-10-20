FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cần thiết cho xử lý ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Sao chép mã nguồn ứng dụng
COPY app ./app
COPY models ./models
COPY README.md .
COPY AI_MODEL_CARD.md .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cpu

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
