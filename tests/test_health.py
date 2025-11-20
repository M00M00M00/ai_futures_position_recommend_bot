from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_returns_running_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Futures Position Recommend Bot is running"}


def test_health_returns_ok_and_model_name():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_model" in body
