# Testing

## 1. Model Evaluation (automated, via train.py)

`train.py` automatically computes on the held-out test set:
- Accuracy, Precision, Recall, F1 Score per class (`classification_report.txt`)
- Confusion Matrix (`confusion_matrix.png`)
- ROC curves + AUC per class, one-vs-rest (`roc_curve.png`)

Run it after training completes:
```bash
cat model/classification_report.txt
```

## 2. Backend unit tests (pytest)

Save as `tests/test_app.py`:

```python
import io
import pytest
from PIL import Image
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color=(100, 180, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_home_page(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_upload_page(client):
    resp = client.get("/upload")
    assert resp.status_code == 200


def test_predict_valid_image(client):
    resp = client.post(
        "/predict",
        data={"image": (_dummy_image_bytes(), "test.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Confidence" in resp.data


def test_predict_no_file(client):
    resp = client.post("/predict", data={}, content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200  # redirected back to upload with a flash message


def test_predict_invalid_extension(client):
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"not an image"), "test.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200


def test_history_page(client):
    resp = client.get("/history")
    assert resp.status_code == 200


def test_dashboard_page(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_search_history(client):
    resp = client.get("/history/search?q=tomato")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_404(client):
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
```

Run with:
```bash
pip install pytest --break-system-packages
pytest tests/ -v
```

## 3. Manual test cases

| # | Test Case | Steps | Expected Result |
|---|---|---|---|
| 1 | Upload valid JPEG | Go to Upload → choose a .jpg → submit | Redirects to Result page with vegetable, status, confidence |
| 2 | Upload unsupported file | Try uploading a .txt or .pdf | Error flash message, stays on Upload page |
| 3 | Upload oversized file (>10MB) | Upload a very large image | 413 error handled, flash message shown |
| 4 | Webcam capture | Start Camera → Capture Photo → Submit | Same result flow as file upload |
| 5 | View history | Go to History after several scans | Table lists all past predictions, newest first |
| 6 | Search history | Type "rotten" in search box | Table filters to only Rotten records via AJAX |
| 7 | Delete history record | Click trash icon → confirm | Record removed from table and DB |
| 8 | Download PDF report | Click "Download Report" on Result page | PDF downloads with image, status, confidence, recommendation |
| 9 | Dashboard stats | Visit Dashboard after several scans | KPIs and per-vegetable bar chart reflect actual DB data |
| 10 | Empty state | Fresh install, no predictions yet | History/Dashboard show friendly empty-state messaging, not errors |
| 11 | 404 handling | Visit an invalid URL | Custom 404 page renders |
| 12 | Demo mode | Run app before training a model | Demo Mode banner shown; predictions still work, clearly labeled as simulated |

## 4. Performance testing (once real model is trained)

- Measure average `prediction_ms` across 50 varied images (mix of
  lighting conditions, vegetable types, fresh and rotten).
- Target: sub-300ms inference on CPU (MobileNetV2 is optimised for this).
- Confirm the dashboard's "Avg. Speed" KPI matches your manual timing.
