import threading
import time
from typing import Any

from app.main import _LazyDockerClient, create_app
from fastapi.testclient import TestClient


class CloseTrackingClient:
    marker = "ready"

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_lazy_docker_client_initializes_once_across_threads() -> None:
    client = CloseTrackingClient()
    factory_calls = 0
    factory_guard = threading.Lock()
    start = threading.Barrier(9)

    def factory() -> CloseTrackingClient:
        nonlocal factory_calls
        with factory_guard:
            factory_calls += 1
        time.sleep(0.05)
        return client

    lazy = _LazyDockerClient(factory=factory)
    results: list[str] = []

    def worker() -> None:
        start.wait(timeout=5)
        results.append(lazy.marker)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert results == ["ready"] * 8
    assert factory_calls == 1


def test_lazy_docker_client_close_is_idempotent_and_does_not_force_initialization() -> None:
    created: list[CloseTrackingClient] = []

    def factory() -> CloseTrackingClient:
        client = CloseTrackingClient()
        created.append(client)
        return client

    never_used = _LazyDockerClient(factory=factory)
    never_used.close()
    never_used.close()
    assert created == []

    initialized = _LazyDockerClient(factory=factory)
    assert initialized.marker == "ready"
    initialized.close()
    initialized.close()
    assert len(created) == 1
    assert created[0].close_calls == 1


def test_app_lifespan_closes_lazy_docker_even_when_context_raises(settings, monkeypatch) -> None:
    close_calls: list[Any] = []
    original_close = _LazyDockerClient.close

    def tracked_close(self: _LazyDockerClient) -> None:
        close_calls.append(self)
        original_close(self)

    monkeypatch.setattr(_LazyDockerClient, "close", tracked_close)

    try:
        with TestClient(create_app(settings)):
            raise RuntimeError("lifespan test")
    except RuntimeError as exc:
        assert str(exc) == "lifespan test"

    assert len(close_calls) == 1
