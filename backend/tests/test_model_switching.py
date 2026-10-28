"""Tests for model availability and switching endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app, AVAILABLE_MODELS


@pytest.fixture
def client():
    return TestClient(app)


def test_get_available_models(client):
    """Test /model/available endpoint returns catalog of supported models."""
    response = client.get("/model/available")
    assert response.status_code == 200
    data = response.json()
    assert "current_model" in data
    assert "current_key" in data
    assert "models" in data
    
    keys = [m["key"] for m in data["models"]]
    assert "tinyllama" in keys
    assert "llama3.2" in keys
    
    # Check that Llama 3.2 model entry has required metadata
    llama_entry = next(m for m in data["models"] if m["key"] == "llama3.2")
    assert "Llama 3.2" in llama_entry["name"]
    assert "3.2" in llama_entry["badge"]
