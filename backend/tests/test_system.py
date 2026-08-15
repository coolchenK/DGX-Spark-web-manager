from app.services.system import SystemService, parse_nvidia_smi, read_os_name


def test_parse_nvidia_smi_preserves_unsupported_unified_memory():
    output = "NVIDIA GB10, 580.173.02, 41, 12.09, [N/A], 0\n"

    result = parse_nvidia_smi(output)

    assert result == [
        {
            "name": "NVIDIA GB10",
            "driver_version": "580.173.02",
            "temperature_c": 41.0,
            "power_w": 12.09,
            "memory_used_bytes": None,
            "utilization_percent": 0.0,
        }
    ]


def test_read_os_name_from_host_os_release(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.4 LTS"\n', encoding="utf-8"
    )

    assert read_os_name(os_release) == "Ubuntu 24.04.4 LTS"


def test_system_service_prefers_host_os_release(tmp_path, monkeypatch):
    os_release = tmp_path / "os-release"
    os_release.write_text('PRETTY_NAME="Ubuntu 24.04.4 LTS"\n', encoding="utf-8")
    service = SystemService(tmp_path, os_release_path=os_release)
    monkeypatch.setattr(service, "_gpu_snapshot", lambda: [])

    assert service.snapshot()["os"] == "Ubuntu 24.04.4 LTS"


def test_system_endpoint_uses_real_service_contract(authenticated_client, monkeypatch):
    expected = {
        "hostname": "gx10-test",
        "architecture": "aarch64",
        "os": "Ubuntu 24.04",
        "kernel": "6.17.0-nvidia",
        "cpu": {"percent": 12.5, "cores": 20},
        "memory": {"total_bytes": 128, "used_bytes": 64, "available_bytes": 64},
        "disk": {"total_bytes": 1000, "used_bytes": 250, "free_bytes": 750},
        "gpus": [],
        "uptime_seconds": 120,
    }
    monkeypatch.setattr(
        authenticated_client.app.state.system_service,
        "snapshot",
        lambda: expected,
    )

    response = authenticated_client.get("/api/system")

    assert response.status_code == 200
    assert response.json() == expected
