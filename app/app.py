from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize app
app = FastAPI(title="Sentiment Analysis API")

# Load model - in a real app we'd do this at startup
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Define input/output models
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    score: float

# Define endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    result = sentiment_analyzer(input_data.text)[0]
    return PredictionOutput(
        sentiment=result["label"],
        score=result["score"]
    )