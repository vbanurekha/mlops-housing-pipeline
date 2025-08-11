from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
import logging
import time
import sqlite3
import os
from typing import Optional
import joblib
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ----------------------------
# Logging setup
# ----------------------------
LOG_PATH = os.getenv("APP_LOG", "app.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mlops_app")

# ----------------------------
# App + metrics
# ----------------------------
app = FastAPI(title="Housing Price Predictor (MLOps)")

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
PRED_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency seconds")

# ----------------------------
# Model loading
# ----------------------------
MODEL = None
MODEL_SRC: Optional[str] = None

def load_model():
    global MODEL, MODEL_SRC
    try:
        MODEL = mlflow.pyfunc.load_model("models:/housing_model/1")
        MODEL_SRC = "mlflow_registry:models:/housing_model/1"
        logger.info("Loaded model from MLflow registry.")
        return
    except Exception as e:
        logger.warning("MLflow registry load failed: %s", e)

    local_path = os.getenv("MODEL_PATH", "models/model.joblib")
    if os.path.exists(local_path):
        try:
            MODEL = joblib.load(local_path)
            MODEL_SRC = f"local_joblib:{local_path}"
            logger.info("Loaded model from %s", local_path)
            return
        except Exception as e:
            logger.exception("Failed to load local model: %s", e)

    logger.error("No model available to load.")
    raise FileNotFoundError("No model available.")

try:
    load_model()
except Exception as e:
    logger.warning("Model not loaded at startup: %s", e)
    MODEL = None

# ----------------------------
# SQLite persistence
# ----------------------------
DB_PATH = os.getenv("PRED_DB", "predictions.db")

def log_prediction_to_db(input_json: dict, prediction: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS preds(timestamp TEXT, input_json TEXT, pred REAL, model_src TEXT)"
        )
        cur.execute(
            "INSERT INTO preds(timestamp, input_json, pred, model_src) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), str(input_json), prediction, MODEL_SRC),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception("DB logging failed: %s", e)

# ----------------------------
# MLflow configuration
# ----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("HousingPriceAPI")
    logger.info(f"MLflow tracking set to {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.warning("MLflow setup failed: %s", e)

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_src": MODEL_SRC}

@app.post("/predict")
async def predict(request: Request):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    payload = await request.json()
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        features = [[
            float(payload["MedInc"]),
            float(payload["HouseAge"]),
            float(payload["AveRooms"]),
            float(payload["AveBedrms"]),
            float(payload["Population"]),
            float(payload["AveOccup"]),
            float(payload["Latitude"]),
            float(payload["Longitude"]),
        ]]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    try:
        pred = MODEL.predict(features)
        pred_val = float(pred[0]) if hasattr(pred, "__iter__") else float(pred)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start
    PRED_LATENCY.observe(latency)
    logger.info("Prediction input=%s output=%s latency=%.4f", payload, pred_val, latency)
    log_prediction_to_db(payload, pred_val)

    # Log to MLflow
    try:
        with mlflow.start_run():
            for k, v in payload.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("prediction", pred_val)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.set_tag("model_source", MODEL_SRC or "unknown")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    return {"prediction": pred_val, "latency": latency, "model_src": MODEL_SRC}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ----------------------------
# HTML UI
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def form_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Housing Price Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f5f7fa; margin: 0; padding: 0; }
            .container { max-width: 600px; margin: 50px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { text-align: center; color: #333; }
            label { display: block; margin-top: 15px; font-weight: bold; }
            input { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; }
            button { margin-top: 20px; padding: 12px; width: 100%; background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 4px; color: #2e7d32; font-weight: bold; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Housing Price Predictor</h2>
            <form id="predictForm">
                <label>Median Income:</label>
                <input type="number" step="any" name="MedInc" required>
                <label>House Age:</label>
                <input type="number" step="any" name="HouseAge" required>
                <label>Average Rooms:</label>
                <input type="number" step="any" name="AveRooms" required>
                <label>Average Bedrooms:</label>
                <input type="number" step="any" name="AveBedrms" required>
                <label>Population:</label>
                <input type="number" step="any" name="Population" required>
                <label>Average Occupancy:</label>
                <input type="number" step="any" name="AveOccup" required>
                <label>Latitude:</label>
                <input type="number" step="any" name="Latitude" required>
                <label>Longitude:</label>
                <input type="number" step="any" name="Longitude" required>
                <button type="submit">Predict</button>
            </form>
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
            document.getElementById('predictForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = {};
                formData.forEach((value, key) => data[key] = parseFloat(value));

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const resultDiv = document.getElementById('result');
                if (res.ok) {
                    const json = await res.json();
                    resultDiv.textContent = `Predicted Price: ${json.prediction.toFixed(2)} | Model: ${json.model_src}`;
                    resultDiv.style.display = 'block';
                } else {
                    resultDiv.textContent = 'Prediction failed.';
                    resultDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
