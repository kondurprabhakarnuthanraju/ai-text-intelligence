# TextSignal AI — AI Text Intelligence

TextSignal AI is an experimental AI text analysis application that classifies text as likely human-written or AI-generated using a machine learning ensemble. It combines a Next.js frontend, a FastAPI backend, and an offline-trained text classification model.

**Live Demo:** https://ai-text-intelligence-app.vercel.app

> **Disclaimer:** AI text detection is inherently unreliable. This application provides an experimental prediction, not proof of authorship. Human-written text may be flagged as AI-generated, and AI-generated text may be classified as human-written.

## Features

- **AI Text Classification:** Analyze text and receive a predicted class with an associated score.
- **Machine Learning:** Uses TF-IDF features and Logistic Regression as the lightweight baseline classifier.
- **Ensemble Preparation:** Includes model and training components for experimenting with ensemble-based text detection.
- **Optional AI Explanations:** Uses LangChain and an OpenAI model to explain predictions in plain English.
- **FastAPI Backend:** Provides an API for predictions, explanations, and health checks.
- **Next.js Frontend:** Offers a web interface for interacting with the classification API.
- **Deployment Support:** Configured for a combined Next.js and Python deployment on Vercel, subject to platform limits.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | Next.js, React, TypeScript, CSS |
| Backend | Python, FastAPI |
| Machine Learning | scikit-learn, TF-IDF, Logistic Regression |
| Training Data | HC3 (Human–ChatGPT Comparison Corpus) |
| AI Explanations | LangChain, OpenAI API (optional) |
| Deployment | Vercel |

## Architecture

```text
                 User
                   |
                   v
          Next.js Frontend
                   |
                   v
             FastAPI API
                   |
                   v
          Text Preprocessing
                   |
                   v
          TF-IDF + Classifier
                   |
                   v
        Prediction and AI Score
                   |
                   v
            Frontend Result

     Optional: LangChain explanation
     based on the existing prediction
```

The language model is used only to explain the classifier's output. It does not determine or override the classification.

## Project Structure

```text
ai-text-intelligence/
├── api/
│   └── index.py
├── app/
│   ├── page.tsx
│   ├── layout.tsx
│   └── globals.css
├── artifacts/
│   └── text_detector.joblib
├── backend/
├── frontend/
├── training/
│   └── train_model.py
├── bert_model.py
├── ensemble_model.py
├── streamlit_app.py
├── requirements.txt
├── package.json
├── next.config.ts
├── tsconfig.json
├── .env.example
├── .gitignore
├── .vercelignore
└── README.md
```

The exact files present in the repository may vary as the project evolves.

## Getting Started

### Prerequisites

- Python 3.11 recommended
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/kondurprabhakarnuthanraju/ai-text-intelligence.git
cd ai-text-intelligence
```

### 2. Set up the Python environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r training/requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

### 3. Train the model

```bash
python training/train_model.py
```

The training script downloads the HC3 dataset, prepares training and evaluation data, trains the classifier, and saves the model artifact.

Expected output:

```text
artifacts/text_detector.joblib
```

Training dependencies are maintained separately from the API dependencies.

If a compatible trained artifact is already available, you can skip training. Ensure the artifact exists and matches the inference code before starting the API.

### 4. Start the backend

Install the API dependencies if you have not already done so:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn api.index:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open another terminal in the repository root:

```bash
npm install
```

Create a `.env.local` file in the repository root:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Start the development server:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

## API Usage

The API accepts JSON requests through `POST /api`.

### Health check

Request:

```json
{
  "action": "health"
}
```

Use this endpoint to check the API's health response.

### Predict text

Request:

```json
{
  "action": "predict",
  "text": "Enter the text you want to analyze."
}
```

The response format depends on the fields returned by the deployed API implementation.

### Explain a prediction

Request:

```json
{
  "action": "explain",
  "text": "Enter the text you want to analyze.",
  "prediction": {
    "label": "Human-written",
    "ai_score": 0.23
  }
}
```

The explanation describes the supplied prediction and its limitations. The example prediction above is illustrative, not an actual model result.

## Optional LangChain Integration

To enable AI-generated explanations, configure the following environment variables in your local environment or deployment settings:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

- `OPENAI_API_KEY` must remain secret.
- Never expose the API key through a `NEXT_PUBLIC_` environment variable.
- The model name is configurable.
- Without the API key, the application can use its deterministic explanation fallback, provided the backend is configured accordingly.

## Dataset and Training

The baseline classifier uses the HC3 dataset, which contains human responses and ChatGPT-generated responses.

The training workflow separates data by question before flattening the answers to reduce the risk of related answers appearing in both training and test sets.

For more rigorous evaluation, consider testing on independent datasets such as RAID, which includes multiple language models, domains, and adversarial examples.

Review dataset and source-dataset licensing terms before redistributing training data or derived artifacts.

## Evaluation and Limitations

Accuracy alone is not sufficient to evaluate an AI text detector. Useful metrics include:

- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive and false-negative rates
- Performance on unseen datasets and newer language models

The model may perform differently across domains, writing styles, languages, and text lengths. A prediction score should not be interpreted as a calibrated probability unless calibration has been evaluated.

Do not use this experimental classifier as the sole basis for academic misconduct allegations, hiring decisions, or other consequential judgments.

## Deployment

The repository is configured for a Vercel deployment that combines the Next.js frontend with a Python API function.

General deployment steps:

1. Push the repository to GitHub.
2. Import the repository into Vercel.
3. Configure the project root and build settings for the combined application.
4. Ensure the Python function and its dependencies are supported by the selected runtime.
5. Confirm the trained model artifact is present and included in the deployment bundle.
6. Configure optional environment variables in Vercel.
7. Deploy and test the health check, prediction endpoint, and frontend.

**Deployment note:** Python runtime compatibility, serverless bundle-size limits, execution limits, and cold starts can affect deployment. If the model or dependencies exceed current platform limits, deploy the Python API separately and configure `NEXT_PUBLIC_API_URL` to point to that service.

## Future Improvements

- Evaluate the model on independent and adversarial datasets.
- Compare different feature extraction and classification approaches.
- Improve calibration and communicate prediction uncertainty.
- Add tests for API endpoints and preprocessing.
- Improve robustness across domains and text lengths.
- Explore ensemble models and additional evaluation metrics.

## License

Add a license file to the repository before specifying reuse or redistribution terms.

---

**TextSignal AI** — an experimental project exploring machine learning, natural language processing, and responsible AI text analysis.
