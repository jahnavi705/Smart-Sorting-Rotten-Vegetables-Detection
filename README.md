# 🥕 Smart Sorting — Rotten Vegetable Detection System

A deep-learning-powered web application that detects whether a vegetable is
**Fresh** or **Rotten** from a photo, identifies the vegetable type, and
recommends a sorting action — built with Flask, TensorFlow/Keras
(MobileNetV2 transfer learning), OpenCV, and SQLite.

> **Status note**: This package ships with the **complete, working
> application code** — Flask backend, full frontend, database layer,
> training script, and documentation. It does **not** ship a pretrained
> `.h5` model file, since that requires training on real image data (see
> [Training](#training) below). Until you train the model, the app runs in
> **Demo Mode**, clearly labeled in the UI, so you can test the entire
> upload → predict → history → report pipeline immediately.

---

## Features

- 📤 Upload an image or 📸 capture directly from your webcam
- 🧠 MobileNetV2-based CNN classifies vegetable type + Fresh/Rotten status
- 📊 Confidence score and inference time shown for every prediction
- 🗂️ Full prediction history with search and delete
- 📈 Analytics dashboard (fresh/rotten ratio, per-vegetable breakdown)
- 📄 One-click downloadable PDF inspection report
- 🎨 Responsive, custom-designed UI (Bootstrap 5 + custom theme)
- 🗄️ SQLite persistence, no external DB server required

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | TensorFlow/Keras, MobileNetV2 (transfer learning) |
| Computer Vision | OpenCV |
| Frontend | HTML, CSS, JavaScript, Bootstrap 5 |
| Database | SQLite |
| Reports | ReportLab (PDF generation) |
| Deployment | Localhost → Render / Railway |

## Project Structure

```
SmartSorting/
├── app.py                     # Flask application & routes
├── train.py                   # Model training (transfer learning pipeline)
├── predict.py                 # Inference module (with Demo Mode fallback)
├── requirements.txt
│
├── model/
│   ├── class_indices.json     # class index -> label mapping
│   ├── README.md               # notes on the trained model file
│   └── smart_sorting_model.h5  # <- YOU generate this by running train.py
│
├── database/
│   └── database.py            # SQLite schema + CRUD helpers
│
├── static/
│   ├── css/style.css
│   ├── js/main.js              # upload/drag-drop logic
│   ├── js/camera.js            # webcam capture logic
│   ├── uploads/                # saved user images
│   └── reports/                # generated PDF reports
│
├── templates/                 # Jinja2 HTML templates
│
├── utils/
│   ├── preprocessing.py       # image resize/normalise helpers
│   ├── report_generator.py    # PDF report builder
│   └── logger.py              # centralised logging
│
├── dataset/                   # train/val/test folders (empty skeleton — you populate)
│
└── docs/                      # full project documentation & diagrams
    ├── 01_dataset_guide.md
    ├── 02_project_report.md    # abstract, architecture, diagrams, etc.
    ├── 03_testing.md
    └── 04_deployment.md
```

## Installation

```bash
# 1. Clone / unzip the project, then cd into it
cd SmartSorting

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick Start (Demo Mode — no training required)

```bash
python app.py
```
Visit `http://localhost:5000`. Upload any image and you'll get a fully
working (simulated) prediction, history entry, dashboard update, and
downloadable PDF report — useful for reviewing the whole UX before you
invest time in training.

## Dataset

See **`docs/01_dataset_guide.md`** for:
- Recommended Kaggle datasets
- Folder structure (`dataset/train|val|test/<Vegetable>_<Fresh|Rotten>/`)
- Cleaning, balancing, augmentation, and train/val/test splitting scripts

## Training

Once your dataset is organised under `dataset/train`, `dataset/val`,
`dataset/test`:

```bash
python train.py
```

This runs a two-phase transfer-learning pipeline (frozen head, then
fine-tuning), and produces:
- `model/smart_sorting_model.h5` — the trained model
- `model/class_indices.json` — regenerated to match your actual folders
- `model/training_history.png` — accuracy/loss curves
- `model/confusion_matrix.png`, `model/roc_curve.png`, `model/classification_report.txt`

**Tip**: training is much faster with a GPU. If you don't have one
locally, upload the project to **Google Colab**, enable a GPU runtime
(Runtime → Change runtime type → GPU), and run `train.py` there — then
download the generated `model/` files back into your local project.

Once `model/smart_sorting_model.h5` exists, restart `python app.py` — it
will automatically detect the file and switch from Demo Mode to real
predictions. No code changes needed.

## Running the App

```bash
python app.py
```

## Screenshots

*(Add screenshots here after running the app locally — e.g. Home page,
Upload page, Result page, History table, Dashboard.)*

## Documentation

Full academic-style documentation — Abstract, Problem Statement,
Literature Survey, System Architecture, Flowchart, DFD, Use Case Diagram,
ER Diagram, Sequence Diagram, Activity Diagram, Testing, Results, Future
Scope, Conclusion, and IEEE References — lives in `docs/02_project_report.md`.

## Future Improvements

- Physical sorting via conveyor belt + servo motor (Arduino/Raspberry Pi)
- Multi-user accounts and role-based access
- Batch/video-stream inspection for continuous conveyor lines
- TensorFlow Lite edge deployment
- Severity grading beyond binary Fresh/Rotten

## License

Provided for academic/educational use.
