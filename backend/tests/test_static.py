from app.main import create_app
from fastapi.testclient import TestClient


def test_spa_fallback_does_not_capture_api_routes(settings, tmp_path):
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<main>manager</main>", encoding="utf-8")
    app_settings = settings.model_copy(
        update={"static_dir": static_dir, "auto_discovery": False}
    )

    with TestClient(create_app(app_settings)) as client:
        page = client.get("/dashboard")
        api = client.get("/api/not-a-real-route")
        gateway = client.get("/v1/not-a-real-route")

    assert page.status_code == 200
    assert "manager" in page.text
    assert api.status_code == 404
    assert gateway.status_code == 404
    assert api.headers["content-type"].startswith("application/json")
