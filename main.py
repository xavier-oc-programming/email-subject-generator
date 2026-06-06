import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import MODEL_DIR, NUM_SUGGESTIONS, T5_MODEL_DIR

_INDEX_PATH = Path('templates/index.html')
_CONFIG_PATH = MODEL_DIR / 't5_config.json'

_model_file = T5_MODEL_DIR / 'model.safetensors'
MODEL_LOADED = _model_file.exists() and _model_file.stat().st_size > 1_000_000

if MODEL_LOADED:
    from generator import generate_multiple

app = FastAPI(
    title="Email Subject Line Generator",
    description="Generates professional email subject line suggestions using T5-small fine-tuned on AESLC.",
    version="1.0.0"
)


class GenerateRequest(BaseModel):
    body_text: str = Field(..., min_length=10, max_length=500)
    n: int = Field(5, ge=1, le=10)
    temperature: float = Field(None, ge=0.1, le=2.0)


class SubjectSuggestion(BaseModel):
    subject: str
    temperature: float
    style: str


class GenerateResponse(BaseModel):
    suggestions: list[SubjectSuggestion]
    body_preview: str
    model_info: str
    n_generated: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: int | None = None
    latent_dim: int | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_INDEX_PATH.read_text())


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_LOADED,
        vocab_size=None,
        latent_dim=None,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    temp_range = (0.5, 0.9)
    if request.temperature is not None:
        t = request.temperature
        temp_range = (max(0.1, t - 0.2), min(2.0, t + 0.2))

    suggestions_raw = generate_multiple(request.body_text, n=request.n, temperature_range=temp_range)

    suggestions = [
        SubjectSuggestion(
            subject=s['subject'],
            temperature=s['temperature'],
            style=s['style']
        )
        for s in suggestions_raw
    ]

    model_info = "T5-small | Fine-tuned on AESLC | 3 epochs"
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        model_info = (
            f"T5-small | Fine-tuned on AESLC | "
            f"epochs={cfg.get('epochs', 3)} | "
            f"train_size={cfg.get('train_size', 'N/A')}"
        )

    return GenerateResponse(
        suggestions=suggestions,
        body_preview=request.body_text[:100],
        model_info=model_info,
        n_generated=len(suggestions)
    )


@app.get("/api/model-info")
def model_info():
    if not _CONFIG_PATH.exists():
        return {"status": "model not trained", "model_loaded": MODEL_LOADED}
    with open(_CONFIG_PATH) as f:
        return json.load(f)


@app.get("/api/examples")
def examples():
    return [
        {
            "body": "Following up on our discussion about the Q3 budget proposal. Please review the attached document and confirm the next steps.",
            "description": "Budget follow-up"
        },
        {
            "body": "I wanted to schedule a meeting to discuss the upcoming product launch and align on the timeline for all deliverables.",
            "description": "Meeting request"
        },
        {
            "body": "Please find attached the updated project report for this quarter. Let me know if you have any questions or need clarification.",
            "description": "Project update"
        },
        {
            "body": "We are pleased to invite you to our annual conference taking place next month. Please confirm your attendance at your earliest convenience.",
            "description": "Conference invitation"
        },
        {
            "body": "I wanted to check in on the status of the client proposal we discussed last week. Has there been any feedback from their team?",
            "description": "Client proposal check-in"
        }
    ]
