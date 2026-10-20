from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Kiểm tra endpoint GET /health."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["UP", "DEGRADED"]
    assert "modelLoaded" in data
    assert isinstance(data["modelLoaded"], bool)
    assert "supportedFruits" in data
    assert isinstance(data["supportedFruits"], list)
    assert len(data["supportedFruits"]) > 0
    assert "Táo" in data["supportedFruits"]
    assert "Cam sành" in data["supportedFruits"]
