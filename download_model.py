# download_model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# This script is run ONLY during the Docker build process.
# Its purpose is to download the model files and save them
# inside the Docker image, so they don't need to be downloaded at runtime.

model_name = 'tuner007/pegasus_paraphrase'

print(f"Downloading model: {model_name}")
AutoTokenizer.from_pretrained(model_name)
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model download complete.")
