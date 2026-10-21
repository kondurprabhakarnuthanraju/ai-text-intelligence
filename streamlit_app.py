import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "text_detector.joblib"

st.set_page_config(
    page_title="TextSignal — AI Text Analysis",
    page_icon="✳",
    layout="wide",
)

st.title("✳ TextSignal")
st.caption("EXPERIMENTAL TEXT ANALYSIS · ML + optional LangChain explanation")
st.markdown(
    "Explore patterns in a piece of writing with a lightweight machine-learning "
    "classifier. **A score is not proof of authorship.**"
)

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def explain_result(text: str, prediction: dict[str, Any]) -> str:
    # Streamlit secrets are preferred for hosted deployments; environment variables
    # remain supported for local use.
    api_key = os.getenv("OPENAI_API_KEY", "")
    try:
        if not api_key and "OPENAI_API_KEY" in st.secrets:
            api_key = str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass

    if not api_key:
        return (
            f"The classifier labeled this text as {prediction['label']} with an AI score "
            f"of {prediction['ai_score']:.1%}. This is the model's output, not a verified "
            "probability of authorship. The model learned patterns from the HC3 dataset "
            "and may be wrong on different topics, writing styles, edited text, or newer "
            "models. Treat this as a preliminary signal, not evidence."
        )

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You explain an experimental text-classification result. Do not independently "
                "decide whether the text is AI-generated. Do not claim authorship can be proven. "
                "Do not invent linguistic evidence. Explain the supplied label and score, state "
                "uncertainty and limitations, and write 3-5 concise sentences. Do not ask for or "
                "repeat the user's text."
            ),
            (
                "human",
                "Classifier label: {label}\nAI score: {score}\nText length: {length} characters."
            ),
        ])
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0.2,
            timeout=20,
            max_retries=1,
        )
        response = llm.invoke(prompt.format_messages(
            label=prediction["label"],
            score=f"{prediction['ai_score']:.1%}",
            length=len(text),
        ))
        return str(response.content)
    except Exception:
        return (
            "The classifier returned a preliminary estimate, not proof of authorship. "
            "False positives and false negatives are possible, especially for short, edited, "
            "translated, or out-of-domain text. The optional explanation service is unavailable; "
            "the original prediction is unchanged."
        )

with st.sidebar:
    st.subheader("About this demo")
    st.write("Model: TF-IDF + Logistic Regression")
    st.write("Maximum input: 20,000 characters")
    st.info(
        "For research and demonstration only. Human text may be flagged, and generated "
        "text may be missed."
    )

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

text = st.text_area(
    "Paste your text",
    height=260,
    max_chars=20000,
    placeholder="Paste at least 40 characters of English text here...",
)
st.caption(f"{len(text)} / 20,000 characters")

if st.button("Analyze text", type="primary", use_container_width=False):
    cleaned = text.strip()
    if len(cleaned) < 40:
        st.warning("Please provide at least 40 characters for a more meaningful prediction.")
    elif len(cleaned) > 20000:
        st.error("Text exceeds the 20,000-character limit.")
    else:
        try:
            model = bundle["model"]
            probabilities = model.predict_proba([cleaned])[0]
            ai_score = float(probabilities[1])
            prediction = {
                "label": "AI-generated" if ai_score >= 0.5 else "Human-written",
                "ai_score": ai_score,
                "human_score": float(probabilities[0]),
            }

            st.divider()
            st.subheader("Analysis snapshot")
            left, right = st.columns(2)
            left.metric("Predicted label", prediction["label"])
            right.metric("AI score", f"{prediction['ai_score']:.1%}")
            st.progress(max(0.0, min(1.0, prediction["ai_score"])))
            score1, score2 = st.columns(2)
            score1.metric("AI score", f"{prediction['ai_score']:.1%}")
            score2.metric("Human score", f"{prediction['human_score']:.1%}")

            metrics = bundle.get("metrics", {})
            if metrics:
                with st.expander("Model evaluation metrics"):
                    st.json(metrics)

            st.subheader("Explanation")
            with st.spinner("Preparing a cautious explanation..."):
                st.write(explain_result(cleaned, prediction))

            st.warning(
                "This experimental estimate is not proof of authorship. The model can make "
                "mistakes, especially on short, edited, translated, or out-of-domain text."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.caption("Built with Streamlit · scikit-learn · joblib · optional LangChain")
