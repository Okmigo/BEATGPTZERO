FROM pytorch/pytorch:2.1.0-cpu

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn transformers

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
