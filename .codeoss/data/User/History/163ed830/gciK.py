# ✅ Step-by-step breakdown follows. Let me help you fix & finalize your pipeline.

### 🔍 High-Level Goal Recap:
# Embed all PDFs from Google Drive folder -> store vectors in embeddings.db -> upload to GCS -> GPT API queries it in real time.

### ❌ Current Setup Problems:
# 1. embed.py doesn't download or process Google Drive files at all.
# 2. There's no watcher or sync to check when new files are added to Drive.
# 3. No PDF parsing or real embedding generation in embed.py
# 4. Database schema used in embed.py (`embeddings(id, hash)`) doesn't match main.py (`documents(id, title, url, text, embedding)`)

# ✅ Next steps (Step-by-step fix):
# --- We'll start with fixed embed.py that:
# (1) Downloads PDF files from Google Drive folder
# (2) Embeds using SentenceTransformer
# (3) Stores results in a SQLite DB with correct schema
# (4) Uploads the DB to GCS

# === embed.py (fixed) ===
import os
import sqlite3
import PyMuPDF
import fitz  # PyMuPDF
import hashlib
import tempfile
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from google.cloud import storage

# Config
FOLDER_ID = "1XtKZcNHAjCf_FNPJMPOwT8QfqbdD9uvW"
EMBED_BUCKET = os.getenv("EMBED_BUCKET", "mystical-gpt-bucket")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "embeddings.db"

# === Authenticate GDrive ===
def get_gdrive_service():
    creds = service_account.Credentials.from_service_account_file("service_account.json")
    return build("drive", "v3", credentials=creds)

# === Download PDF ===
def download_pdf(file_id, filename):
    service = get_gdrive_service()
    request = service.files().get_media(fileId=file_id)
    with open(filename, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

# === Extract text from PDF ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

# === Main embedding logic ===
def embed_pdfs():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    creds = service_account.Credentials.from_service_account_file("service_account.json")
    drive_service = build("drive", "v3", credentials=creds)

    query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    files = drive_service.files().list(q=query, fields="files(id, name)").execute().get("files", [])

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS documents")
    c.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            text TEXT,
            embedding TEXT
        )
    """)

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        print(f"[📄] Processing {file_name}")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            download_pdf(file_id, tmp.name)
            text = extract_text_from_pdf(tmp.name)
            embedding = model.encode(text).tolist()
            embedding_str = ",".join(map(str, embedding))
            url = f"https://drive.google.com/file/d/{file_id}/view"
            c.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", (file_id, file_name, url, text, embedding_str))

    conn.commit()
    conn.close()
    print("[✅] embeddings.db created")

# === Upload DB to GCS ===
def upload_db():
    creds = service_account.Credentials.from_service_account_file("service_account.json")
    client = storage.Client(credentials=creds)
    bucket = client.bucket(EMBED_BUCKET)
    blob = bucket.blob("embeddings.db")
    blob.upload_from_filename(DB_PATH)
    print("[🚀] Uploaded to GCS")

if __name__ == "__main__":
    embed_pdfs()
    upload_db()
