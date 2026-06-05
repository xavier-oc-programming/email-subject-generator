from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_BODY = "Following up on our discussion about the Q3 budget proposal. Please review the attached document and confirm the next steps."


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate():
    response = client.post("/generate", json={"body_text": SAMPLE_BODY, "n": 3})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 3
        for s in data["suggestions"]:
            assert "subject" in s
            assert "style" in s
            assert len(s["subject"]) > 0


def test_generate_too_short():
    response = client.post("/generate", json={"body_text": "hi", "n": 3})
    assert response.status_code == 422


def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200


def test_examples():
    response = client.get("/api/examples")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
