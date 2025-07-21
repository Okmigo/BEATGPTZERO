from fastapi import FastAPI
app = FastAPI()

# Simple test route
@app.get("/")
def root():
    return {"status": "running"}
