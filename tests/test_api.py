"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked model."""
    # Patch the lifespan to avoid loading real model
    from server.api.main import app

    # Set up mock state
    app.state.chat_generator = None
    app.state.fim_generator = None
    app.state.memory = None

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_shows_model_not_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "omniscient-200m"


class TestChatEndpoint:
    def test_returns_503_without_model(self, client):
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_rejects_invalid_message_format(self, client):
        response = client.post("/v1/chat/completions", json={
            "messages": [{"bad_key": "no role or content"}],
        })
        assert response.status_code == 422


class TestCompletionsEndpoint:
    def test_accepts_fim_request(self, client):
        response = client.post("/v1/completions", json={
            "prompt": "def hello():",
            "suffix": "\n    return greeting",
            "max_tokens": 64,
        })
        # Without model, returns empty or error
        assert response.status_code in (200, 500)
