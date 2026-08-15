import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DGX_SECRET_KEY", "test-secret-key-with-at-least-32-characters")
os.environ.setdefault("DGX_ADMIN_PASSWORD", "Test-password-1234")

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'manager.db'}",
        data_dir=tmp_path,
        secret_key="test-secret-key-with-at-least-32-characters",
        admin_password="Test-password-1234",
        allowed_origins="http://testserver",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def authenticated_client(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    assert response.status_code == 200
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return client

