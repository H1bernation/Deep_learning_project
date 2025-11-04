from fastapi import FastAPI

app = FastAPI(title="Skin Analysis API")

@app.get("/")
def read_root():
    return {
        "message": "Skin Analysis API is running!",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}