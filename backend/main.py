import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "text_detector.joblib"

app = FastAPI(
    title="AI Text Intelligence API",
    description="Experimental AI-vs-human text classifier with optional LangChain explanation.",
    version="1.0.0",
)

# Same-origin requests are used in production. This permissive CORS setting is
# helpful for local development; tighten it to your deployed frontend domain if
# you host the API separately.
allowed_origins = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

_model_bundle = None


def get_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact is missing. Run python training/train_model.py and deploy artifacts/text_detector.joblib.",
            )
        try:
            _model_bundle = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model artifact: {exc}")
    return _model_bundle


class RequestBody(BaseModel):
    action: str = Field(default="predict", description="health, predict, or explain")
    text: str = Field(default="", max_length=20000)
    prediction: dict[str, Any] | None = None


@app.get("/")
def root():
    return {"service": "AI Text Intelligence API", "status": "ok", "docs": "/docs"}


@app.post("/")
@app.post("/api")
def handle_request(body: RequestBody):
    action = body.action.lower().strip()

    if action == "health":
        return {
            "status": "ok",
            "model_available": MODEL_PATH.exists(),
            "langchain_configured": bool(os.getenv("OPENAI_API_KEY")),
        }

    if action == "predict":
        text = body.text.strip()
        if len(text) < 40:
            raise HTTPException(status_code=422, detail="Please provide at least 40 characters for a more meaningful prediction.")
        if len(text) > 20000:
            raise HTTPException(status_code=413, detail="Text exceeds the 20,000-character limit.")

        bundle = get_bundle()
        model = bundle["model"]
        probabilities = model.predict_proba([text])[0]
        ai_score = float(probabilities[1])
        label = "AI-generated" if ai_score >= 0.5 else "Human-written"
        return {
            "label": label,
            "ai_score": round(ai_score, 4),
            "human_score": round(float(probabilities[0]), 4),
            "model": "TF-IDF + Logistic Regression",
            "metrics": bundle.get("metrics", {}),
            "caveat": "Experimental estimate, not proof of authorship. Results can be wrong, especially on short, edited, translated, or out-of-domain text.",
        }

    if action == "explain":
        if not body.prediction:
            raise HTTPException(status_code=422, detail="Provide the prediction object returned by the predict action.")
        return {"explanation": explain_with_langchain(body.text, body.prediction)}

    raise HTTPException(status_code=400, detail="action must be one of: health, predict, explain")


def explain_with_langchain(text: str, prediction: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        label = prediction.get("label", "unknown")
        score = prediction.get("ai_score", "unknown")
        return (
            f"The classifier labeled this text as {label} with an AI score of {score}. "
            "This score is the model's output, not a verified probability of authorship. "
            "The model learned patterns from the HC3 dataset and may be wrong on different topics, "
            "writing styles, edited text, or text from newer models. Treat this as a preliminary signal, not evidence."
        )

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant explaining an experimental text-classification result. "
             "Do not independently decide whether the text is AI-generated. Do not claim authorship can be proven. "
             "Do not invent linguistic evidence or pretend to know how a particular author writes. "
             "Explain what the supplied label and score mean, state uncertainty and limitations, and give a short, neutral summary. "
             "Do not reveal or repeat the user's full text."),
            ("human",
             "Classifier label: {label}\n"
             "Classifier AI score: {score}\n"
             "Text length: {length} characters\n"
             "Write a concise explanation in 3-5 sentences. Explicitly say the score is not proof of authorship.")
        ])
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            timeout=20,
            max_retries=1,
        )
        messages = prompt.format_messages(
            label=str(prediction.get("label", "unknown")),
            score=str(prediction.get("ai_score", "unknown")),
            length=len(text or ""),
        )
        response = llm.invoke(messages)
        return str(response.content)
    except Exception:
        # Keep prediction available if the optional explanation provider fails.
        return (
            "The classifier returned a preliminary estimate, not proof of authorship. "
            "The model can make false positives and false negatives, especially on short, edited, translated, "
            "or out-of-domain text. The explanation service is temporarily unavailable; the original prediction is unchanged."
        )
