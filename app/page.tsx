"use client";

import { useMemo, useState } from "react";

type Prediction = {
  label: string;
  ai_score: number;
  human_score: number;
  model: string;
  metrics?: Record<string, unknown>;
  caveat: string;
};

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export default function Home() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [error, setError] = useState("");

  const characterCount = text.length;
  const canAnalyze = useMemo(() => text.trim().length >= 40 && !loading, [text, loading]);

  async function callApi(payload: Record<string, unknown>) {
    const response = await fetch(`${API_URL}/api`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "Request failed. Check the API deployment.");
    }
    return data;
  }

  async function analyze() {
    setLoading(true);
    setError("");
    setPrediction(null);
    setExplanation("");
    try {
      const data = await callApi({ action: "predict", text });
      setPrediction(data as Prediction);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function explain() {
    if (!prediction) return;
    setExplaining(true);
    setError("");
    try {
      const data = await callApi({
        action: "explain",
        text,
        prediction: {
          label: prediction.label,
          ai_score: prediction.ai_score,
          human_score: prediction.human_score,
        },
      });
      setExplanation(data.explanation);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate explanation.");
    } finally {
      setExplaining(false);
    }
  }

  function clearAll() {
    setText("");
    setPrediction(null);
    setExplanation("");
    setError("");
  }

  const aiPercent = prediction ? Math.round(prediction.ai_score * 100) : 0;

  return (
    <main className="page-shell">
      <nav className="nav">
        <a className="brand" href="#"><span className="brand-mark">✳</span> TextSignal</a>
        <span className="nav-tag">ML + LangChain demo</span>
      </nav>

      <section className="hero">
        <div className="eyebrow"><span className="live-dot" /> EXPERIMENTAL TEXT ANALYSIS</div>
        <h1>Understand the signal.<br /><span>Not just the score.</span></h1>
        <p className="hero-copy">
          Explore patterns in a piece of writing with a lightweight machine-learning classifier,
          then use a LangChain-powered assistant to explain the result and its limitations.
        </p>
      </section>

      <section className="workspace">
        <div className="input-panel panel">
          <div className="panel-heading">
            <div>
              <div className="step-label">01 / INPUT</div>
              <h2>Paste your text</h2>
            </div>
            <span className="text-badge">English text</span>
          </div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, 20000))}
            placeholder="Paste a paragraph or a longer passage here. Longer samples generally give the classifier more information to work with..."
            maxLength={20000}
            aria-label="Text to analyze"
          />
          <div className="input-footer">
            <span>{characterCount.toLocaleString()} / 20,000 characters</span>
            <button className="text-button" onClick={clearAll} type="button">Clear text</button>
          </div>
          <button className="primary-button" disabled={!canAnalyze} onClick={analyze} type="button">
            {loading ? <><span className="spinner" /> Analyzing…</> : <>Analyze text <span>↗</span></>}
          </button>
          <p className="microcopy">Your pasted text is sent to the API for prediction. Do not submit confidential or sensitive material.</p>
        </div>

        <div className="result-panel panel">
          <div className="panel-heading">
            <div>
              <div className="step-label">02 / RESULT</div>
              <h2>Analysis snapshot</h2>
            </div>
            <span className="model-chip">TF-IDF · LR</span>
          </div>

          {!prediction && !loading && (
            <div className="empty-state">
              <div className="empty-icon">⌁</div>
              <h3>Your result will appear here</h3>
              <p>Run an analysis to see the model's label and score distribution.</p>
            </div>
          )}
          {loading && (
            <div className="empty-state">
              <span className="big-spinner" />
              <h3>Reading the patterns</h3>
              <p>The saved classifier is processing your text.</p>
            </div>
          )}
          {prediction && (
            <div className="result-content">
              <div className="result-label">PREDICTED LABEL</div>
              <div className="prediction-title">
                <span className={`prediction-icon ${prediction.ai_score >= 0.5 ? "ai" : "human"}`}>
                  {prediction.ai_score >= 0.5 ? "A" : "H"}
                </span>
                <div>
                  <h3>{prediction.label}</h3>
                  <p>Experimental classifier output</p>
                </div>
              </div>
              <div className="score-heading">
                <span>AI-class score</span><strong>{aiPercent}%</strong>
              </div>
              <div className="score-track"><div className="score-fill" style={{ width: `${aiPercent}%` }} /></div>
              <div className="score-legend">
                <span><i className="legend-ai" /> AI score: {(prediction.ai_score * 100).toFixed(1)}%</span>
                <span><i className="legend-human" /> Human score: {(prediction.human_score * 100).toFixed(1)}%</span>
              </div>
              <div className="notice"><strong>Read with care</strong><p>{prediction.caveat}</p></div>
              <button className="secondary-button" disabled={explaining} onClick={explain} type="button">
                {explaining ? "Preparing explanation…" : "Explain this result with LangChain"} <span>↗</span>
              </button>
              {explanation && <div className="explanation"><div className="step-label">LANGCHAIN EXPLANATION</div><p>{explanation}</p></div>}
            </div>
          )}
        </div>
      </section>

      {error && <div className="error-message" role="alert">{error}</div>}

      <section className="feature-row">
        <article><span className="feature-number">01</span><h3>Classify</h3><p>TF-IDF features and Logistic Regression produce the initial model output.</p></article>
        <article><span className="feature-number">02</span><h3>Explain</h3><p>LangChain passes the existing result to an LLM for a cautious, plain-English explanation.</p></article>
        <article><span className="feature-number">03</span><h3>Stay honest</h3><p>A score is not proof. Human writing may be flagged, and generated writing may be missed.</p></article>
      </section>

      <footer>
        <span>Built with Next.js · FastAPI · scikit-learn · LangChain</span>
        <span>For research and demonstration only</span>
      </footer>
    </main>
  );
}
