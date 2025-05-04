from fastapi import FastAPI

app = FastAPI(title="API Tester")

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/predict")
def predict_dummy(text: str = "test"):
    return {"sentiment": "POSITIVE", "score": 0.99}
