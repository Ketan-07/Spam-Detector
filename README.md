# Spam Message Detector

A full-stack spam/ham message classifier: a scikit-learn pipeline
(TF-IDF + Logistic Regression) trained on the classic SMS Spam
Collection dataset, served by a FastAPI backend, with a themed web UI
("Sorting Office") on top.

**🔴 Live app:** https://spam-detector-ud17.onrender.com
**📦 Code:** https://github.com/Ketan-07/Spam-Detector

> Hosted on Render's free tier — the first request after a period of
> inactivity can take 20–30 seconds while the server wakes up. That's
> expected, not a bug.

---

## Project structure

```
spam-detector/
├── app/
│   ├── main.py          # FastAPI app: /api/predict, /api/health, serves static/
│   ├── train_model.py   # Trains the pipeline and saves model/spam_model.pkl
│   └── __init__.py
├── data/
│   └── spam.csv         # Training data (label, message)
├── model/
│   └── spam_model.pkl   # Trained pipeline
├── static/
│   ├── index.html       # UI
│   ├── style.css
│   └── script.js
├── build.sh             # Installs deps and retrains the model at deploy time
├── requirements.txt
└── .gitignore
```

---

## Run it locally

**Requirements:** Python 3.10+

```bash
# from the spam-detector/ folder
python3 -m venv venv
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\Activate.ps1

pip install -r requirements.txt

# train the model (creates model/spam_model.pkl)
python app/train_model.py

# start the app — serves both the API and the UI on one port
python -m uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

### Test the API directly

```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER!! Click here to claim your free prize now!!!"}'
```

Expected response:

```json
{"label": "spam", "is_spam": true, "confidence": 0.87}
```

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: spam detector full-stack app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

---

## Deploy (Render)

This app is deployed on [Render](https://render.com) as a single web
service (backend + frontend together).

1. Push the project to GitHub.
2. Render → **New** → **Web Service** → connect the GitHub repo.
3. Configure:
   - **Language:** Python 3
   - **Build Command:** `bash build.sh`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `PYTHON_VERSION` = `3.11.9`
     (keeps Render on a Python version with pre-built wheels for
     pandas/scikit-learn, avoiding slow/broken source compiles)
4. Instance type: **Free** (fine for testing/demo use).
5. Click **Create Web Service**.

`build.sh` installs dependencies and retrains the model as part of the
build, so the model is always created by the exact same scikit-learn
version that serves it — this avoids version-mismatch errors between
a locally-trained `.pkl` file and the server's installed packages.

```bash
# build.sh
pip install -r requirements.txt
python app/train_model.py
```

Once deployed, Render gives you a public URL
(`https://<your-service-name>.onrender.com`), and auto-redeploys on
every push to `main`.

---

## How it works

- **Model** (`app/train_model.py`): loads `data/spam.csv`, splits
  train/test, fits a `Pipeline(TfidfVectorizer → LogisticRegression)`,
  and saves it with `joblib` to `model/spam_model.pkl`.
- **API** (`app/main.py`): loads the saved pipeline at startup, exposes
  `POST /api/predict`, which returns the predicted label (`spam`/`ham`),
  a boolean `is_spam`, and a `confidence` score from `predict_proba`.
- **UI** (`static/`): a single page that posts your message to
  `/api/predict` and animates the result into a "ham" or "spam" tray.
