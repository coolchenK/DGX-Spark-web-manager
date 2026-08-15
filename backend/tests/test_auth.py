def test_session_status_is_public_and_restores_authenticated_user(client):
    anonymous = client.get("/api/auth/session")
    assert anonymous.status_code == 200
    assert anonymous.json() == {
        "authenticated": False,
        "user": None,
        "csrf_token": None,
    }

    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    restored = client.get("/api/auth/session")

    assert restored.status_code == 200
    assert restored.json() == {
        "authenticated": True,
        "user": {"username": "admin", "role": "admin"},
        "csrf_token": login.json()["csrf_token"],
    }


def test_login_sets_secure_session_and_returns_current_user(client):
    rejected = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong-password"},
    )
    assert rejected.status_code == 401

    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )

    assert response.status_code == 200
    assert response.json()["user"]["username"] == "admin"
    assert response.json()["csrf_token"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert client.get("/api/auth/me").json() == {"username": "admin", "role": "admin"}


def test_protected_mutation_requires_csrf(client):
    login = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "Test-password-1234"},
    )
    assert login.status_code == 200

    response = client.post("/api/keys", json={"name": "SDK"})

    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF token is missing or invalid"


def test_api_key_is_returned_once_and_can_be_revoked(authenticated_client):
    created = authenticated_client.post("/api/keys", json={"name": "Local SDK"})
    assert created.status_code == 201
    body = created.json()
    assert body["key"].startswith("dgx_")
    assert body["prefix"] == body["key"][:12]

    listed = authenticated_client.get("/api/keys")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "Local SDK"
    assert "key" not in listed.json()[0]

    revoked = authenticated_client.delete(f"/api/keys/{body['id']}")
    assert revoked.status_code == 204
    assert authenticated_client.get("/api/keys").json()[0]["revoked_at"] is not None
