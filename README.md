# AI Text Intelligence — Vercel + Python + LangChain

A small portfolio-ready AI text classification app:
- Next.js frontend deployed on Vercel
- Python FastAPI endpoint deployed as a Vercel Python Function
- TF-IDF + Logistic Regression model trained offline on HC3
- Optional LangChain explanation of the prediction and its limitations
- No database and no agent framework complexity

> Important: this is an experimental classifier, not proof of authorship. Human text can be flagged as AI and AI text can be missed.

## Dataset choice

Primary training dataset: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), the Human–ChatGPT Comparison Corpus. It has paired human answers and ChatGPT answers and is manageable for a first version.

For more rigorous robustness evaluation later, use [RAID](https://huggingface.co/datasets/liamdugan/raid). RAID is much larger and includes multiple models, domains, and adversarial attacks, so it is not the best first dataset for a lightweight deployment.

HC3 is licensed CC-BY-SA-4.0 at the dataset card level, but the card notes that source datasets may have stricter terms. Review the dataset card and applicable source licenses before redistribution.

## 1. Train the model locally

Use Python 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r training/requirements.txt
python training/train_model.py
```

This downloads HC3, splits by question row before flattening the answers (to reduce question leakage between train/test), trains the lightweight model, prints metrics, and saves:
`artifacts/text_detector.joblib`

Do not train inside the deployed request handler.

## 2. Run locally

```bash
uvicorn api.index:app --reload
```

API is at `http://127.0.0.1:8000`. The FastAPI docs are at `/docs`.

In a second terminal:

```bash
npm install
npm run dev
```

Frontend is at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` in `.env.local` for local API calls.

## 3. Add LangChain explanation (optional)

Set these environment variables on the Python service / Vercel project:
- `OPENAI_API_KEY` — secret; never put it in a `NEXT_PUBLIC_` variable
- `OPENAI_MODEL` — optional; defaults to `gpt-4o-mini`

The `explain` action uses LangChain to explain the classifier output in plain English and explicitly state limitations. If the key is absent, the API returns a safe deterministic explanation instead.

The LLM does **not** make or change the classification. It only explains the existing model output.

## 4. Deploy to Vercel

1. Push this folder to a GitHub repository.
2. Import the repository into Vercel.
3. In Vercel Project Settings, configure the frontend root as the repository root.
4. Ensure Python Functions are enabled for the project. The Python API lives in `api/index.py`; the root `requirements.txt` contains API-only dependencies. Training-only dependencies are isolated in `training/requirements.txt`.
5. Train the model locally and include `artifacts/text_detector.joblib` in the repository. The `.gitignore` does not ignore this artifact. Check its size before deploying; serverless function bundle limits apply.
6. Add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in Vercel Environment Variables if you want the LangChain explanation.
7. Deploy and test `/api` using the health action, then use the website.

### Deployment caveat

Vercel's Python runtime and serverless bundle limits can change. If the saved model or dependencies exceed current limits, keep the frontend on Vercel and deploy the Python API on a Python-friendly host instead. A free serverless deployment is not a guarantee of uninterrupted 24/7 warm execution.

## API contract

POST `/api` with one of these JSON bodies:

```json
{"action":"predict","text":"Paste text here"}
```

```json
{"action":"explain","text":"Paste text here","prediction":{"label":"Human-written","ai_score":0.23}}
```

```json
{"action":"health"}
```

## Project structure

```text
ai-text-intelligence-vercel/
├── api/index.py
├── artifacts/.gitkeep
├── training/train_model.py
├── app/page.tsx
├── app/layout.tsx
├── app/globals.css
├── package.json
├── tsconfig.json
├── next.config.ts
├── requirements.txt
├── .env.example
└── .gitignore
```

## Evaluation notes

Accuracy alone is insufficient. Review precision, recall, F1, confusion matrix, and performance on a separate dataset/domain. HC3 is mostly human answers compared with ChatGPT answers, so results may not generalize to every writing style, language, or newer LLM. Avoid describing the returned probability as a calibrated certainty unless calibration has been measured.
