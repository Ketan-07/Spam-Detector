# Spam Message Detector

A full-stack spam/ham message classifier: a scikit-learn pipeline
(TF-IDF + Logistic Regression) trained on the classic SMS Spam
Collection dataset, served by a FastAPI backend, with a small
themed web UI ("Sorting Office") on top.

```
spam-detector/
├── app/
│   ├── main.py          # FastAPI app: /api/predict, /api/health, serves static/
│   ├── train_model.py   # Trains the pipeline and saves model/spam_model.pkl
│   └── __init__.py
├── data/
│   └── spam.csv         # Training data (label, message)
├── model/
│   └── spam_model.pkl   # Trained pipeline (already included, ready to use)
├── static/
│   ├── index.html        # UI
│   ├── style.css
│   └── script.js
├── requirements.txt
├── render.yaml           # One-click config for deploying on Render
└── .gitignore
```

The model is already trained and included (`model/spam_model.pkl`), so
you can run the app immediately without retraining. Retrain any time
with `python app/train_model.py` (e.g. after changing the dataset).

---

## 1. Run it locally

**Requirements:** Python 3.10+

```bash
# from the spam-detector/ folder
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# (optional — a trained model is already included)
python app/train_model.py

# start the app (serves both the API and the UI)
uvicorn app.main:app --reload --port 8000
```

Open **[http://localhost:8000**](https://spam-detector-ud17.onrender.com/) in your browser — that's the whole app,
UI and API together on one port.

### Test the API directly

```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER!! Click here to claim your free prize now!!!"}'
```

Expected response shape:

```json
{"label": "spam", "is_spam": true, "confidence": 0.87}
```

---

## 2. Push to GitHub

```bash
cd spam-detector

git init
git add .
git commit -m "Initial commit: spam detector full-stack app"

# create a new repo on GitHub first (via github.com or `gh repo create`), then:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> The dataset, trained model, and code contain no secrets or API keys,
> so it's safe to push as a public repo.

---

## 3. Make it live (deploy)

### Option A — Render (recommended, free tier, one repo = whole app)

1. Push the project to GitHub (step 2 above).
2. Go to [render.com](https://render.com) → **New** → **Web Service** →
   connect your GitHub repo.
3. Render will detect `render.yaml` automatically and configure:
   - Build command: `pip install -r requirements.txt && python app/train_model.py`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**. First deploy takes a couple of minutes.
5. Your app is live at `https://<your-service-name>.onrender.com`.

(No `render.yaml`? You can fill these same two commands into Render's
UI manually when creating the service.)

### Option B — Railway

1. Push to GitHub.
2. [railway.app](https://railway.app) → **New Project** → **Deploy from
   GitHub repo**.
3. Set the start command to:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add a build step / run once: `python app/train_model.py` (or just
   commit `model/spam_model.pkl`, which is already included, so this
   is optional).
5. Deploy — Railway gives you a public URL.

### Option C — Any Docker host (Fly.io, Google Cloud Run, etc.)

Minimal `Dockerfile` you can add if you prefer containers:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python app/train_model.py
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t spam-detector .
docker run -p 8000:8000 spam-detector
```

---

## How it works

- **Model** (`app/train_model.py`): loads `data/spam.csv`, splits
  train/test, fits a `Pipeline(TfidfVectorizer → LogisticRegression)`,
  and saves it with `joblib` to `model/spam_model.pkl`. This mirrors
  the original notebook's approach — same features, same model.
- **API** (`app/main.py`): loads the saved pipeline once at startup,
  exposes `POST /api/predict` which returns the predicted label
  (`spam`/`ham`), a boolean `is_spam`, and a `confidence` score from
  `predict_proba`.
- **UI** (`static/`): a single page that posts your message to
  `/api/predict` and animates the result into a "ham" or "spam" tray.

## Notes / next steps

- The original notebook also called a Hugging Face LLM to explain each
  prediction in plain English. That's left out of this app on purpose —
  the notebook had an API token hardcoded in it, which should never be
  committed to a public repo. If you want that feature back, add it as
  a separate endpoint that reads the token from an environment variable
  (`os.environ["HF_TOKEN"]`), and set that variable in your hosting
  provider's dashboard (Render → Environment) rather than in code.
- To retrain on a different/larger dataset, just replace
  `data/spam.csv` (keep the `v1`/`v2` column names, or update
  `train_model.py` to match new column names) and rerun
  `python app/train_model.py`.
