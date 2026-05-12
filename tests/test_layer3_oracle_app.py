from __future__ import annotations

from fastapi.testclient import TestClient

from cascade_defect.layer3_gpt4o import app as layer3_app


class _StubPrediction:
    defect_class = "uncertain"
    confidence = 0.13
    reasoning = "image is too corrupted to classify confidently"


def test_oracle_uncertain_path(monkeypatch):
    def _stub_predict(_image_path, _seed_dir, *, domain=None):
        return _StubPrediction(), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    monkeypatch.setattr(layer3_app, "predict", _stub_predict)
    client = TestClient(layer3_app.app)

    response = client.post(
        "/predict",
        files={"file": ("corrupted.jpg", b"not-a-real-image-but-mocked", "image/jpeg")},
        data={"domain": "metal"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "uncertain"
    assert body["class"] == "uncertain"
    assert body["domain"] == "metal"
