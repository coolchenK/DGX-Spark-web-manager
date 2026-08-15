def test_huggingface_token_is_encrypted_and_applied(authenticated_client):
    initial = authenticated_client.get("/api/settings")
    assert initial.status_code == 200
    assert initial.json()["huggingface"]["token_configured"] is False
    assert "token" not in initial.json()["huggingface"]

    updated = authenticated_client.patch(
        "/api/settings/huggingface",
        json={"token": "hf_test-secret-token-123456"},
    )

    assert updated.status_code == 200
    assert updated.json()["token_configured"] is True
    assert authenticated_client.app.state.huggingface_service.token == "hf_test-secret-token-123456"
    with authenticated_client.app.state.database.session_factory() as db:
        from app.models import SecretSetting

        setting = db.get(SecretSetting, "huggingface_token")
        assert "hf_test-secret-token-123456" not in setting.encrypted_value


def test_spa_is_served_for_client_routes(settings, tmp_path):
    from app.main import create_app
    from fastapi.testclient import TestClient

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>DGX Manager SPA</body></html>")
    settings.static_dir = static_dir

    with TestClient(create_app(settings)) as client:
        response = client.get("/deployments")

    assert response.status_code == 200
    assert "DGX Manager SPA" in response.text
