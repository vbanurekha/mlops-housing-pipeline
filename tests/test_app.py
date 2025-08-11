from fastapi.testclient import TestClient
from src.app import app  #  FastAPI app is in app.py

client = TestClient(app)

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<html>" in response.text.lower()

def test_health():
    """Test a health check endpoint (if available)."""
    response = client.get("/health")
    assert response.status_code in [200, 404]
