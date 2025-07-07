FROM python:3.11-slim

WORKDIR /app

RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir='./hf_cache')"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=3000
EXPOSE 3000

CMD ["python", "rewriter.py"]
