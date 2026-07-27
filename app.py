"""
app.py
------
Main Flask application for the Smart Sorting Rotten Vegetable Detection System.

Routes:
    GET  /                  Home page
    GET  /about             About the project
    GET  /upload            Upload / webcam capture page
    POST /predict           Handle image upload or webcam capture, run prediction, save to DB
    GET  /result/<id>       Show a specific prediction result
    GET  /history           View all prediction history
    POST /history/delete/<id>   Delete a history record
    GET  /history/search    Search history (AJAX)
    GET  /dashboard         Analytics dashboard
    GET  /contact           Contact page
    GET  /report/<id>       Download PDF report for a prediction
"""

import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_file,
    abort,
)
from werkzeug.utils import secure_filename

from database import database as db
import predict as predictor
from utils.report_generator import generate_pdf_report, get_recommendation
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# App configuration
# ------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["REPORT_FOLDER"] = os.path.join("static", "reports")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload size

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------------
# Startup: init DB + load model once
# ------------------------------------------------------------------
db.init_db()
predictor.load_model()


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    stats = db.get_dashboard_stats()
    return render_template("home.html", stats=stats, demo_mode=predictor.is_demo_mode())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html", demo_mode=predictor.is_demo_mode())


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handles two input types:
      1. Standard file upload (request.files['image'])
      2. Webcam capture sent as a Blob (request.files['image'] too — the JS
         wraps the captured frame in a FormData under the same key)
    """
    if "image" not in request.files:
        flash("No image file received. Please choose a file or capture from webcam.", "danger")
        return redirect(url_for("upload_page"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("upload_page"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP.", "danger")
        return redirect(url_for("upload_page"))

    # Save with a unique filename to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(save_path)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        flash("Could not save the uploaded image. Please try again.", "danger")
        return redirect(url_for("upload_page"))

    try:
        result = predictor.predict_image(image_path=save_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        flash(f"Prediction failed: {e}", "danger")
        return redirect(url_for("upload_page"))

    relative_image_path = os.path.join("uploads", unique_name).replace("\\", "/")

    record_id = db.insert_prediction(
        vegetable=result["vegetable"],
        status=result["status"],
        confidence=result["confidence"],
        image_path=relative_image_path,
        prediction_ms=result["prediction_ms"],
    )

    return redirect(url_for("result_page", prediction_id=record_id))


@app.route("/result/<int:prediction_id>")
def result_page(prediction_id):
    records = db.get_all_predictions(limit=1000)
    record = next((r for r in records if r["id"] == prediction_id), None)
    if record is None:
        abort(404)

    recommendation = get_recommendation(record["status"], record["vegetable"])
    return render_template(
        "result.html",
        record=record,
        recommendation=recommendation,
        demo_mode=predictor.is_demo_mode(),
    )


@app.route("/history")
def history_page():
    records = db.get_all_predictions()
    return render_template("history.html", records=records)


@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
def delete_history(prediction_id):
    deleted = db.delete_prediction(prediction_id)
    if deleted:
        flash("Record deleted successfully.", "success")
    else:
        flash("Record not found.", "warning")
    return redirect(url_for("history_page"))


@app.route("/history/search")
def search_history():
    query = request.args.get("q", "").strip()
    if not query:
        records = db.get_all_predictions()
    else:
        records = db.search_predictions(query)
    return jsonify(records)


@app.route("/dashboard")
def dashboard_page():
    stats = db.get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


@app.route("/report/<int:prediction_id>")
def download_report(prediction_id):
    records = db.get_all_predictions(limit=1000)
    record = next((r for r in records if r["id"] == prediction_id), None)
    if record is None:
        abort(404)

    recommendation = get_recommendation(record["status"], record["vegetable"])
    image_full_path = os.path.join("static", record["image_path"]) if record["image_path"] else None

    report_filename = f"report_{prediction_id}_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(app.config["REPORT_FOLDER"], report_filename)

    generate_pdf_report(
        output_path=report_path,
        image_path=image_full_path,
        vegetable_name=record["vegetable"],
        status=record["status"],
        confidence=record["confidence"],
        prediction_time_ms=record["prediction_ms"] or 0,
        recommendation=recommendation,
    )

    return send_file(report_path, as_attachment=True, download_name=f"SmartSorting_Report_{prediction_id}.pdf")


# ------------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum upload size is 10MB.", "danger")
    return redirect(url_for("upload_page"))


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    return render_template("500.html"), 500


if __name__ == "__main__":
    # debug=True is fine for local development; set to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
