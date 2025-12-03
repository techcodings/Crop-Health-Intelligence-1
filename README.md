# Crop Health Intelligence – Advanced Flask Demo

This project is a more advanced prototype implementation of your **Crop Health Intelligence** feature.

It includes:

- A Perceiver-style multimodal encoder for:
  - Satellite indices (NDVI/EVI/SAVI/MSAVI – synthetic but seasonally structured)
  - Weather time series
  - Static terrain/soil/irrigation features
  - Regional agricultural statistics
- UNet-like stress map generator for vegetation stress visualization
- LSTM-based temporal forecaster for crop health indices
- RL-style policy/value head for irrigation / fertilization / scouting decisions
- Management zone segmentation using k-means on the stress map
- Heuristic diagnosis of likely stress drivers (water / disease / irrigation issues)
- A **Flask** web UI with Bootstrap for interacting with the engine
- A JSON REST endpoint `/api/analyze` for programmatic integration

> ⚠️ All data in `backend/data/data_sources.py` is synthetic.  
> Replace it with real EO / weather / soil / irrigation APIs for production.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Flask app

From the project root:

```bash
python app.py
```

Flask will start on `http://127.0.0.1:5000/` by default.

## Usage

- Open the root URL in your browser.
- Enter a **Field ID**, latitude, longitude, and optional notes/photos.
- Click **Run Analysis**.
- You'll see:
  - A vegetation stress heatmap
  - Management zones with per-zone recommendations
  - Summary health indicators
  - A short-term health forecast
  - Alerts and likely stress drivers
  - Action recommendations from the policy head

For API usage, send a JSON `POST` to `/api/analyze`:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"field_id": "field-001", "lat": 10.0, "lon": 77.0}'
```

This will return the full model output (stress map, forecast, probabilities, etc.) as JSON.
