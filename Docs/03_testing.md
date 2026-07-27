# Deployment Guide

## 1. Localhost (development)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First time only) Train the model — see docs/01_dataset_guide.md
python train.py

# 4. Run the app
python app.py
```
Visit `http://localhost:5000`. Flask's built-in dev server (`debug=True`)
is fine for local testing but **must not** be used in production — see
step 2 below.

## 2. Production-style local run (gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```
`-w 4` runs 4 worker processes. Since MobileNetV2 inference is CPU-bound
and the model is loaded once per worker process, tune worker count to your
machine's CPU cores (roughly `2 x cores + 1` is a common starting point,
though for a model-serving app, start lower, e.g. `2-4`, and load-test).

## 3. Deploying to Render

1. Push your project to a GitHub repository (make sure `model/*.h5` is
   either committed via Git LFS or hosted externally — see note below on
   large files).
2. On [render.com](https://render.com), create a **New Web Service** and
   connect your GitHub repo.
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
   - **Environment**: Python 3.11+
4. Add an environment variable `SECRET_KEY` with a random secure value.
5. Deploy. Render will build and give you a public URL.

**Note on the trained model file**: `.h5` model files are often 10-50MB,
which exceeds GitHub's default 100MB soft limits in aggregate with other
assets. Use [Git LFS](https://git-lfs.github.com/) to track it:
```bash
git lfs install
git lfs track "model/*.h5"
git add .gitattributes model/smart_sorting_model.h5
git commit -m "Add trained model via Git LFS"
```

## 4. Deploying to Railway

1. Push to GitHub as above.
2. On [railway.app](https://railway.app), create a **New Project** →
   **Deploy from GitHub repo**.
3. Railway auto-detects Python. Set the **Start Command** under Settings:
   `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
4. Add a `SECRET_KEY` environment variable under the Variables tab.
5. Railway provisions a public domain automatically after deploy.

## 5. Persistent storage note (important for both platforms)

Render and Railway's default filesystems are **ephemeral** — files written
at runtime (uploaded images, `history.db`, generated PDF reports) can be
wiped on redeploy or restart. For a course/academic demo this is usually
fine, but for anything longer-lived:
- Move `history.db` to a managed Postgres instance (both platforms offer
  one) and update `database/database.py` accordingly (swap `sqlite3` for
  `psycopg2` or use SQLAlchemy for portability).
- Move uploaded images to an object store (e.g. Cloudflare R2, AWS S3,
  Render Disks) instead of the local `static/uploads/` folder.

## 6. Environment variables checklist

| Variable | Purpose | Required |
|---|---|---|
| `SECRET_KEY` | Flask session/flash message signing | Yes, in production |
| `PORT` | Set automatically by Render/Railway | Auto |
