import os
import sqlite3
import hashlib
from flask import jsonify
from google.cloud import storage

# === CONFIGURATION ===
TMP_DB_PATH = "/tmp/embeddings.db"
BUCKET_NAME = os.environ.get("EMBED_BUCKET", "mystical-gpt-bucket")
GCS_BLOB_NAME = "embeddings.db"  # Flat structure

def generate_embeddings(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS embeddings (id TEXT, hash TEXT)")
    for i in range(5):
        hash_val = hashlib.md5(str(i).encode()).hexdigest()
        c.execute("INSERT INTO embeddings (id, hash) VALUES (?, ?)", (str(i), hash_val))
    conn.commit()
    conn.close()

def upload_to_bucket(local_path: str, bucket_name: str, dest_blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    print(f"[✅] Uploaded {local_path} to gs://{bucket_name}/{dest_blob_name}")

def main(request):
    try:
        print("[🚀] Starting embedAndDeploy function")
        print(f"[🛠️] Generating embeddings at {TMP_DB_PATH}")
        generate_embeddings(TMP_DB_PATH)

        if not os.path.exists(TMP_DB_PATH):
            raise FileNotFoundError(f"embeddings.db not found at {TMP_DB_PATH}")

        upload_to_bucket(TMP_DB_PATH, BUCKET_NAME, GCS_BLOB_NAME)

        return jsonify({
            "status": "success",
            "gcs_uri": f"gs://{BUCKET_NAME}/{GCS_BLOB_NAME}"
        })

    except Exception as e:
        print(f"[❌] Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500
