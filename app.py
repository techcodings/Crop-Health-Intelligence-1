import os
import uuid
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

from backend.main_service import CropHealthEngine
from backend.utils.config import STATIC_GEN_DIR

app = Flask(__name__)

# --- Make sure static/generated exists ---
try:
    STATIC_GEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] STATIC_GEN_DIR = {STATIC_GEN_DIR}", flush=True)
except Exception as e:
    print("[INIT] Error creating STATIC_GEN_DIR:", e, flush=True)

# --- Load engine once at startup ---
try:
    engine = CropHealthEngine()
    print("[INIT] CropHealthEngine initialised", flush=True)
except Exception as e:
    import traceback
    print("[INIT] Error initialising CropHealthEngine:", e, flush=True)
    traceback.print_exc()
    engine = None


def _save_heatmap(stress_map: np.ndarray, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(stress_map, cmap="RdYlGn_r")
    fig.colorbar(im, ax=ax, label="Stress level")
    ax.set_title("Field Vegetation Stress")
    ax.axis("off")

    path = STATIC_GEN_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    # will be used as url_for('static', filename=heatmap_path)
    return f"generated/{filename}"


def _save_forecast(forecast: np.ndarray, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(forecast) + 1), forecast, marker="o")
    ax.set_xlabel("Future timestep")
    ax.set_ylabel("Health index (normalized)")
    ax.set_title("Forecasted Crop Health Trend")
    ax.grid(True)

    path = STATIC_GEN_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"generated/{filename}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    import traceback

    try:
        field_id = request.form.get("field_id", "field-001")
        lat = float(request.form.get("lat", "10.0"))
        lon = float(request.form.get("lon", "77.0"))
        notes = request.form.get("notes", "")

        # photos accepted but ignored for now
        photos = request.files.getlist("photos")

        if engine is None:
            return "Server error: engine not loaded.", 500

        print(f"[ANALYZE] field_id={field_id}, lat={lat}, lon={lon}", flush=True)

        # heavy work here
        result = engine.run_field(field_id, lat, lon)

        stress = np.array(result["stress_map"])
        forecast = np.array(result["forecast_indices"])

        # save plots
        run_id = uuid.uuid4().hex[:8]
        heatmap_file = f"stress_{run_id}.png"
        forecast_file = f"forecast_{run_id}.png"

        heatmap_path = _save_heatmap(stress, heatmap_file)
        forecast_path = _save_forecast(forecast, forecast_file)

        context = {
            "field_id": field_id,
            "lat": lat,
            "lon": lon,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "heatmap_path": heatmap_path,
            "forecast_path": forecast_path,
            "summary": result["summary"],
            "best_action": result["best_action"],
            "alerts": result["alerts"],
            "policy_probs": result["policy_probs"],
            "policy_value": result["policy_value"],
            "zones": result["zones"],
            "stress_causes": result["stress_causes"],
        }

        print("[ANALYZE] success, rendering results.html", flush=True)
        return render_template("results.html", **context)

    except Exception as e:
        print("[ANALYZE] ERROR:", e, flush=True)
        traceback.print_exc()
        return "Internal server error while analyzing field.", 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    import traceback

    try:
        data = request.json or {}
        field_id = data.get("field_id", "field-001")
        lat = float(data.get("lat", 10.0))
        lon = float(data.get("lon", 77.0))

        if engine is None:
            return jsonify({"error": "Engine not loaded"}), 500

        print(f"[API] analyze field_id={field_id}, lat={lat}, lon={lon}", flush=True)
        result = engine.run_field(field_id, lat, lon)
        return jsonify(result)

    except Exception as e:
        print("[API] ERROR:", e, flush=True)
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
