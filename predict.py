"""
predict.py
----------
Loads the trained model (if available) and exposes a single function,
`predict_image()`, that app.py calls for every uploaded/captured image.

DEMO MODE
=========
If `model/smart_sorting_model.h5` does not exist yet (i.e. you haven't run
train.py on a real dataset), this module automatically falls back to a
clearly-labeled "Demo Mode" that returns a plausible-looking but SIMULATED
prediction. This lets you test the entire Flask app, database, history, and
PDF report pipeline end-to-end before you've trained a real model.
Once you drop a real `smart_sorting_model.h5` into `model/`, this module
will detect it automatically and switch to real predictions — no code
changes required.
"""

import os
import json
import time
import random

import numpy as np

from utils.preprocessing import (
    read_image_from_path,
    read_image_from_bytes,
    preprocess_for_model,
)
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "smart_sorting_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

_model = None
_class_indices = None
_demo_mode = False


def _load_class_indices():
    global _class_indices
    with open(CLASS_INDICES_PATH, "r") as f:
        raw = json.load(f)
    # JSON keys are always strings; convert back to int for indexing
    _class_indices = {int(k): v for k, v in raw.items()}
    return _class_indices


def load_model():
    """
    Load the Keras model once (lazy singleton pattern) so we don't reload
    it from disk on every single request — that would be very slow.
    """
    global _model, _demo_mode

    _load_class_indices()

    if not os.path.exists(MODEL_PATH):
        _demo_mode = True
        logger.warning(
            f"No trained model found at {MODEL_PATH}. "
            f"Starting in DEMO MODE — predictions will be simulated. "
            f"Run train.py on a real dataset to enable real predictions."
        )
        return None

    # Lazy import: keep TensorFlow out of the import path entirely when
    # running in demo mode, so the app can still start on machines without
    # TensorFlow installed (useful for quickly demoing just the UI).
    from tensorflow.keras.models import load_model as keras_load_model

    logger.info(f"Loading trained model from {MODEL_PATH} ...")
    _model = keras_load_model(MODEL_PATH)
    _demo_mode = False
    logger.info("Model loaded successfully.")
    return _model


def is_demo_mode() -> bool:
    return _demo_mode


def _parse_label(raw_label: str):
    """
    Class labels are stored like 'Tomato_Fresh' / 'Potato_Rotten'.
    Split into (vegetable_name, status).
    """
    parts = raw_label.rsplit("_", 1)
    if len(parts) == 2:
        vegetable, status = parts
    else:
        vegetable, status = raw_label, "Unknown"
    return vegetable, status.capitalize()


def _simulated_prediction():
    """Generate a plausible fake prediction for Demo Mode."""
    vegetables = ["Tomato", "Potato", "Onion", "Carrot", "Brinjal",
                  "Cabbage", "Cauliflower", "Chilli", "Cucumber", "Capsicum"]
    vegetable = random.choice(vegetables)
    status = random.choice(["Fresh", "Rotten"])
    confidence = round(random.uniform(82.0, 99.0), 2)
    return vegetable, status, confidence


def predict_image(image_path: str = None, image_bytes: bytes = None):
    """
    Run a prediction on either a saved image path OR raw image bytes
    (used for webcam captures that aren't saved to disk first).

    Returns a dict:
        {
            "vegetable": str,
            "status": "Fresh" | "Rotten",
            "confidence": float (0-100),
            "prediction_ms": float,
            "demo_mode": bool
        }
    """
    if _class_indices is None:
        _load_class_indices()

    start = time.time()

    if image_path:
        img_rgb = read_image_from_path(image_path)
    elif image_bytes:
        img_rgb = read_image_from_bytes(image_bytes)
    else:
        raise ValueError("predict_image requires either image_path or image_bytes")

    if _demo_mode or _model is None:
        # NOTE: we deliberately do NOT call preprocess_for_model() here — it
        # depends on TensorFlow's MobileNetV2 preprocessing, and demo mode is
        # designed to work even on machines without TensorFlow installed
        # (e.g. testing just the Flask UI before setting up the DL environment).
        # We still touch img_rgb (already decoded above) so image reading is
        # genuinely exercised end-to-end.
        _ = img_rgb.shape
        vegetable, status, confidence = _simulated_prediction()
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"[DEMO MODE] Simulated prediction: {vegetable} - {status} ({confidence}%)"
        )
        return {
            "vegetable": vegetable,
            "status": status,
            "confidence": confidence,
            "prediction_ms": round(elapsed_ms, 1),
            "demo_mode": True,
        }

    # ---- Real model inference ----
    batch = preprocess_for_model(img_rgb)
    preds = _model.predict(batch, verbose=0)[0]  # shape: (num_classes,)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx]) * 100

    raw_label = _class_indices.get(class_idx, "Unknown_Unknown")
    vegetable, status = _parse_label(raw_label)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"Prediction: {vegetable} - {status} ({confidence:.2f}%) in {elapsed_ms:.1f}ms")

    return {
        "vegetable": vegetable,
        "status": status,
        "confidence": round(confidence, 2),
        "prediction_ms": round(elapsed_ms, 1),
        "demo_mode": False,
    }
