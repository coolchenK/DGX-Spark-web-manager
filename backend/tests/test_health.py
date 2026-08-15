from app.main import create_app
from fastapi.testclient import TestClient


def test_health_reports_service_and_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DGX_DATABASE_URL", f"sqlite:///{tmp_path / 'manager.db'}")
    monkeypatch.setenv("DGX_SECRET_KEY", "test-secret-key-with-at-least-32-characters")
    monkeypatch.setenv("DGX_ADMIN_PASSWORD", "Test-password-1234")

    with TestClient(create_app()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "dgx-spark-web-manager",
        "database": "ok",
    }
