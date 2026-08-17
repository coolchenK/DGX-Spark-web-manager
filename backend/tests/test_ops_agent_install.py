from __future__ import annotations

import configparser
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install-ops-agent.sh"
UNINSTALLER = ROOT / "scripts" / "uninstall-ops-agent.sh"
MANAGER_INSTALLER = ROOT / "scripts" / "install.sh"
SERVICE_UNIT = ROOT / "deploy" / "systemd" / "dgx-spark-ops-agent.service"
SOCKET_UNIT = ROOT / "deploy" / "systemd" / "dgx-spark-ops-agent.socket"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read_string(_read(path))
    return parser


def _bash() -> str:
    executable = shutil.which("bash")
    if executable is None and os.name == "nt":
        candidates = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/usr/bin/bash.exe",
        )
        executable = next((os.fspath(path) for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("bash is unavailable on this platform")
    return executable


def _run_bash(
    script: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), os.fspath(script), *arguments],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _shell_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if os.name == "nt" and re.match(r"^[A-Za-z]:/", resolved):
        return f"/{resolved[0].lower()}{resolved[2:]}"
    return resolved


def _fake_path(fake_bin: Path) -> str:
    if os.name == "nt":
        return f"{_shell_path(fake_bin)}:/usr/bin:/bin"
    return f"{fake_bin}{os.pathsep}{os.environ['PATH']}"


def _fake_agent_system(
    tmp_path: Path,
    *,
    enabled: bool,
    active: bool,
    health_results: tuple[str, ...],
    fail_first_reload: bool = False,
) -> tuple[Path, Path, Path, Path]:
    real_install = shutil.which("install")
    real_python = shutil.which("python3")
    real_stat = shutil.which("stat")
    if not all((real_install, real_python, real_stat)):
        pytest.skip("required POSIX tools are unavailable")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    state = tmp_path / "systemd-state"
    state.mkdir()
    log = tmp_path / "systemctl.log"
    health = tmp_path / "health-results"
    health.write_text("\n".join(health_results) + "\n", encoding="utf-8")
    for kind in ("socket", "service"):
        (state / f"{kind}.enabled").write_text(
            "enabled" if enabled else "disabled",
            encoding="ascii",
        )
        (state / f"{kind}.active").write_text(
            "active" if active else "inactive",
            encoding="ascii",
        )
    if fail_first_reload:
        (state / "fail-reload-once").touch()

    _write_executable(
        fake_bin / "install",
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "args=()",
                "while [[ $# -gt 0 ]]; do",
                "  case \"$1\" in -o|-g) shift 2 ;; *) args+=(\"$1\"); shift ;; esac",
                "done",
                f'exec "{Path(real_install).as_posix()}" "${{args[@]}}"',
                "",
            )
        ),
    )
    _write_executable(fake_bin / "chown", "#!/bin/sh\nexit 0\n")
    _write_executable(fake_bin / "getent", "#!/bin/sh\nprintf 'dgx-spark-ops:x:2345:\\n'\n")
    _write_executable(fake_bin / "groupadd", "#!/bin/sh\nexit 0\n")
    _write_executable(fake_bin / "openssl", "#!/bin/sh\nprintf '%064d\\n' 0\n")
    _write_executable(
        fake_bin / "stat",
        "\n".join(
            (
                "#!/bin/sh",
                "if [ \"${1:-}\" = \"-c\" ]; then",
                "  printf 'root:dgx-spark-ops:640\\n'",
                "  exit 0",
                "fi",
                f'exec "{Path(real_stat).as_posix()}" "$@"',
                "",
            )
        ),
    )
    _write_executable(
        fake_bin / "python3",
        "\n".join(
            (
                "#!/bin/sh",
                "if [ \"${1:-}\" = \"-\" ] && [ \"$#\" -eq 3 ]; then",
                "  cat >/dev/null",
                f'  result="$(head -n 1 "{health.as_posix()}")"',
                f'  tail -n +2 "{health.as_posix()}" > "{health.as_posix()}.next"',
                f'  mv "{health.as_posix()}.next" "{health.as_posix()}"',
                "  [ \"$result\" = pass ] || exit 9",
                "  printf 'Agent health check passed\\n'",
                "  exit 0",
                "fi",
                f'exec "{Path(real_python).as_posix()}" "$@"',
                "",
            )
        ),
    )
    _write_executable(
        fake_bin / "systemctl",
        "\n".join(
            (
                "#!/bin/sh",
                f'STATE="{state.as_posix()}"',
                f'UNIT_DIR="{(tmp_path / "root/etc/systemd/system").as_posix()}"',
                f'printf "%s\\n" "$*" >> "{log.as_posix()}"',
                "command=$1",
                "shift",
                "case \"$command\" in",
                "  is-enabled|is-active)",
                "    [ -e \"$STATE/query-transport-failure\" ] && exit 55",
                "    [ \"${1:-}\" = --quiet ] && shift",
                "    unit=$1",
                "    case \"$unit\" in *.socket) kind=socket ;; *) kind=service ;; esac",
                "    if [ ! -e \"$UNIT_DIR/$unit\" ]; then",
                "      printf 'not-found\\n'",
                "      exit 4",
                "    fi",
                "    suffix=${command#is-}",
                "    value=$(cat \"$STATE/$kind.$suffix\")",
                "    printf '%s\\n' \"$value\"",
                "    [ -e \"$STATE/query-rc-mismatch\" ] && exit 22",
                "    case \"$command:$value\" in",
                "      is-enabled:enabled|is-enabled:enabled-runtime) exit 0 ;;",
                "      is-enabled:static|is-active:active) exit 0 ;;",
                "      is-enabled:disabled) exit 1 ;;",
                "      is-active:inactive) exit 3 ;;",
                "      *) exit 23 ;;",
                "    esac",
                "    ;;",
                "  daemon-reload)",
                "    if [ -e \"$STATE/fail-reload-once\" ]; then",
                "      rm -f \"$STATE/fail-reload-once\"",
                "      exit 17",
                "    fi",
                "    ;;",
                "  enable|disable)",
                "    if [ \"$command\" = enable ] && [ -e \"$STATE/fail-enable\" ]; then",
                "      rm -f \"$STATE/fail-enable\"; exit 32",
                "    fi",
                "    if [ \"$command\" = disable ] && [ -e \"$STATE/fail-disable\" ]; then",
                "      rm -f \"$STATE/fail-disable\"; exit 33",
                "    fi",
                "    now=false",
                "    runtime=false",
                "    [ \"${1:-}\" = --now ] && { now=true; shift; }",
                "    [ \"${1:-}\" = --runtime ] && { runtime=true; shift; }",
                "    value=disabled",
                "    if [ \"$command\" = enable ]; then",
                "      value=enabled; $runtime && value=enabled-runtime",
                "    fi",
                "    for unit in \"$@\"; do",
                "      case \"$unit\" in *.socket) kind=socket ;; *) kind=service ;; esac",
                "      printf '%s' \"$value\" > \"$STATE/$kind.enabled\"",
                "      if $now; then",
                "        active=inactive; [ \"$command\" = enable ] && active=active",
                "        printf '%s' \"$active\" > \"$STATE/$kind.active\"",
                "      fi",
                "    done",
                "    ;;",
                "  start|stop|restart)",
                "    if [ \"$command\" = stop ] && [ -e \"$STATE/fail-stop\" ]; then",
                "      rm -f \"$STATE/fail-stop\"; exit 31",
                "    fi",
                "    if [ \"$command\" = start ] && [ -e \"$STATE/fail-start\" ]; then",
                "      rm -f \"$STATE/fail-start\"; exit 34",
                "    fi",
                "    if [ \"$command\" = restart ] && [ -e \"$STATE/fail-restart\" ]; then",
                "      rm -f \"$STATE/fail-restart\"; exit 35",
                "    fi",
                "    [ \"$command\" = stop ] && [ -e \"$STATE/sticky-stop\" ] && exit 0",
                "    value=inactive; [ \"$command\" != stop ] && value=active",
                "    for unit in \"$@\"; do",
                "      case \"$unit\" in *.socket) kind=socket ;; *) kind=service ;; esac",
                "      printf '%s' \"$value\" > \"$STATE/$kind.active\"",
                "    done",
                "    ;;",
                "esac",
                "exit 0",
                "",
            )
        ),
    )
    return fake_bin, state, log, health


def _agent_test_env(fake_bin: Path, install_root: Path) -> dict[str, str]:
    return {
        "PATH": _fake_path(fake_bin),
        "DGX_OPS_AGENT_TESTING": "1",
        "DGX_OPS_AGENT_TEST_ARCH": "aarch64",
        "DGX_OPS_AGENT_TEST_EUID": "0",
        "DGX_OPS_AGENT_INSTALL_ROOT": _shell_path(install_root),
    }


def _seed_old_agent(install_root: Path) -> tuple[Path, Path, Path]:
    package = install_root / "usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent"
    service = install_root / "etc/systemd/system/dgx-spark-ops-agent.service"
    socket_unit = install_root / "etc/systemd/system/dgx-spark-ops-agent.socket"
    package.mkdir(parents=True)
    (package / "old-agent.txt").write_text("old-package", encoding="utf-8")
    service.parent.mkdir(parents=True)
    service.write_text("old-service", encoding="utf-8")
    socket_unit.write_text("old-socket", encoding="utf-8")
    service.chmod(0o640)
    socket_unit.chmod(0o600)
    key = install_root / "etc/dgx-spark-manager/ops-agent.key"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("0" * 64 + "\n", encoding="ascii")
    return package, service, socket_unit


def test_systemd_socket_contract() -> None:
    unit = _unit(SOCKET_UNIT)

    assert unit["Socket"] == {
        "ListenStream": "/run/dgx-spark-manager/ops-agent.sock",
        "SocketUser": "root",
        "SocketGroup": "dgx-spark-ops",
        "SocketMode": "0660",
        "RemoveOnStop": "true",
    }
    assert unit["Install"]["WantedBy"] == "sockets.target"


def test_systemd_service_contract_and_recovery_fallback() -> None:
    unit = _unit(SERVICE_UNIT)
    service = unit["Service"]

    assert service["ExecStart"] == "/usr/bin/python3 -m dgx_ops_agent.server --systemd"
    assert service["Environment"] == "PYTHONPATH=/usr/local/lib/dgx-spark-ops-agent"
    assert service["User"] == "root"
    assert service["Group"] == "root"
    assert service["UMask"] == "0027"
    assert service["NoNewPrivileges"] == "true"
    assert service["PrivateTmp"] == "true"
    assert service["KillMode"] == "control-group"
    assert "ProtectSystem" not in service


def test_compose_connects_manager_to_agent_with_supplementary_group() -> None:
    compose = _read(ROOT / "compose.yaml")

    manager = compose.split("  manager:", maxsplit=1)[1]
    assert re.search(r"(?m)^\s{6}- \"\$\{OPS_AGENT_GID(?::-\d+)?\}\"\s*$", manager)
    assert (
        "- /run/dgx-spark-manager:/run/dgx-spark-manager" in manager
    )
    assert (
        "- /etc/dgx-spark-manager/ops-agent.key:/run/secrets/ops-agent.key:ro"
        in manager
    )
    expected_environment = {
        "DGX_OPS_AGENT_SOCKET": "/run/dgx-spark-manager/ops-agent.sock",
        "DGX_OPS_AGENT_KEY_FILE": "/run/secrets/ops-agent.key",
    }
    for name, value in expected_environment.items():
        assert re.search(rf"(?m)^\s{{6}}{name}: {re.escape(value)}\s*$", manager)
    assert "DGX_OPS_AGENT_CONNECT_TIMEOUT_SECONDS:" in manager
    assert "DGX_OPS_AGENT_READ_TIMEOUT_SECONDS:" in manager
    assert "DGX_OPS_AGENT_OUTPUT_LIMIT_BYTES:" in manager


def test_installer_has_preview_and_strict_apply_contract() -> None:
    script = _read(INSTALLER)

    assert "set -Eeuo pipefail" in script
    assert "main \"$@\"" in script
    assert "--apply" in script
    assert "Unknown argument" in script
    assert "aarch64" in script and "arm64" in script
    assert "effective root" in script
    assert "sudo ./scripts/install-ops-agent.sh --apply" in script
    assert "openssl rand -hex 32" in script
    assert "0640" in script
    assert "daemon-reload" in script
    assert "systemctl enable dgx-spark-ops-agent.socket" in script
    assert "systemctl start dgx-spark-ops-agent.socket" in script
    assert "agent.health" in script
    assert "recovery" not in script.lower() or "rollback" in script.lower()


def test_installer_rejects_unknown_arguments() -> None:
    result = _run_bash(INSTALLER, "--unexpected")

    assert result.returncode != 0
    assert "Unknown argument" in result.stderr


def test_installer_preview_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    install_root = tmp_path / "root"
    result = _run_bash(
        INSTALLER,
        env={
            "DGX_OPS_AGENT_TESTING": "1",
            "DGX_OPS_AGENT_INSTALL_ROOT": _shell_path(install_root),
            "DGX_OPS_AGENT_TEST_ARCH": "aarch64",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Preview only" in result.stdout
    assert not install_root.exists()


def test_apply_refuses_non_arm64_and_non_root() -> None:
    wrong_arch = _run_bash(
        INSTALLER,
        "--apply",
        env={"DGX_OPS_AGENT_TESTING": "1", "DGX_OPS_AGENT_TEST_ARCH": "x86_64"},
    )
    assert wrong_arch.returncode != 0
    assert "Unsupported architecture" in wrong_arch.stderr

    non_root = _run_bash(
        INSTALLER,
        "--apply",
        env={
            "DGX_OPS_AGENT_TESTING": "1",
            "DGX_OPS_AGENT_TEST_ARCH": "aarch64",
            "DGX_OPS_AGENT_TEST_EUID": "1000",
        },
    )
    assert non_root.returncode != 0
    assert "effective root" in non_root.stderr
    assert "sudo ./scripts/install-ops-agent.sh --apply" in non_root.stderr


def test_apply_is_idempotent_and_preserves_key_and_jobs_with_fake_system(
    tmp_path: Path,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX ownership semantics")
    fake_bin, _, log, _ = _fake_agent_system(
        tmp_path,
        enabled=False,
        active=False,
        health_results=("pass", "pass"),
    )

    install_root = tmp_path / "root"
    jobs = install_root / "var/lib/dgx-spark-ops-agent/jobs"
    jobs.mkdir(parents=True)
    marker = jobs / "existing.json"
    marker.write_text("preserve", encoding="utf-8")
    env = _agent_test_env(fake_bin, install_root)

    first = _run_bash(INSTALLER, "--apply", env=env)
    assert first.returncode == 0, first.stderr
    key = install_root / "etc/dgx-spark-manager/ops-agent.key"
    original_key = key.read_bytes()
    second = _run_bash(INSTALLER, "--apply", env=env)

    assert second.returncode == 0, second.stderr
    assert key.read_bytes() == original_key
    assert len(original_key.strip()) == 64
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert (install_root / "usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent/server.py").is_file()
    systemctl_calls = log.read_text(encoding="utf-8")
    assert "daemon-reload" in systemctl_calls
    assert "enable --now dgx-spark-ops-agent.socket" in systemctl_calls


def test_upgrade_health_failure_restores_old_agent_units_and_systemd_state(
    tmp_path: Path,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, log, health = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("fail", "pass"),
    )
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)

    result = _run_bash(
        INSTALLER,
        "--apply",
        env=_agent_test_env(fake_bin, install_root),
    )

    assert result.returncode == 9
    assert (package / "old-agent.txt").read_text(encoding="utf-8") == "old-package"
    assert not (package / "server.py").exists()
    assert service.read_text(encoding="utf-8") == "old-service"
    assert socket_unit.read_text(encoding="utf-8") == "old-socket"
    assert service.stat().st_mode & 0o777 == 0o640
    assert socket_unit.stat().st_mode & 0o777 == 0o600
    for kind in ("socket", "service"):
        assert (state / f"{kind}.enabled").read_text() == "enabled"
        assert (state / f"{kind}.active").read_text() == "active"
    calls = log.read_text(encoding="utf-8")
    assert calls.count("daemon-reload") == 2
    assert "start dgx-spark-ops-agent.socket" in calls
    assert "start dgx-spark-ops-agent.service" in calls
    assert health.read_text(encoding="utf-8") == ""
    assert not list((package.parent).glob(".transaction.*"))
    assert not list((package.parent).glob(".previous.*"))


def test_first_install_health_failure_removes_new_artifacts_and_disables_units(
    tmp_path: Path,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, _, health = _fake_agent_system(
        tmp_path,
        enabled=False,
        active=False,
        health_results=("fail", "pass"),
    )
    install_root = tmp_path / "root"

    result = _run_bash(
        INSTALLER,
        "--apply",
        env=_agent_test_env(fake_bin, install_root),
    )

    assert result.returncode == 9
    assert not (install_root / "usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent").exists()
    assert not (install_root / "etc/systemd/system/dgx-spark-ops-agent.service").exists()
    assert not (install_root / "etc/systemd/system/dgx-spark-ops-agent.socket").exists()
    for kind in ("socket", "service"):
        assert (state / f"{kind}.enabled").read_text() == "disabled"
        assert (state / f"{kind}.active").read_text() == "inactive"
    assert health.read_text(encoding="utf-8") == "pass\n"


def test_systemctl_mid_install_failure_uses_the_same_complete_rollback(
    tmp_path: Path,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, log, health = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
        fail_first_reload=True,
    )
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)

    result = _run_bash(
        INSTALLER,
        "--apply",
        env=_agent_test_env(fake_bin, install_root),
    )

    assert result.returncode == 17
    assert (package / "old-agent.txt").is_file()
    assert service.read_text(encoding="utf-8") == "old-service"
    assert socket_unit.read_text(encoding="utf-8") == "old-socket"
    for kind in ("socket", "service"):
        assert (state / f"{kind}.enabled").read_text() == "enabled"
        assert (state / f"{kind}.active").read_text() == "active"
    assert log.read_text(encoding="utf-8").count("daemon-reload") == 2
    assert health.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("failure", ["fail-stop", "sticky-stop"])
def test_install_stop_failures_exit_before_writing_artifacts(
    tmp_path: Path,
    failure: str,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, _, _ = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
    )
    (state / failure).touch()
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)

    result = _run_bash(INSTALLER, "--apply", env=_agent_test_env(fake_bin, install_root))

    assert result.returncode != 0
    assert (package / "old-agent.txt").read_text() == "old-package"
    assert service.read_text() == "old-service"
    assert socket_unit.read_text() == "old-socket"
    assert not (install_root / "var/lib/dgx-spark-ops-agent/jobs").exists()
    assert not list(package.parent.glob(".transaction.*"))
    assert not list(package.parent.glob(".previous.*"))


@pytest.mark.parametrize(
    ("marker", "expected_status"),
    [("fail-enable", 32), ("fail-start", 34), ("fail-restart", 35)],
)
def test_new_systemd_activation_failure_rolls_back_without_committing_backup(
    tmp_path: Path,
    marker: str,
    expected_status: int,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, _, _ = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
    )
    (state / marker).touch()
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)

    result = _run_bash(INSTALLER, "--apply", env=_agent_test_env(fake_bin, install_root))

    assert result.returncode == expected_status
    assert (package / "old-agent.txt").is_file()
    assert service.read_text() == "old-service"
    assert socket_unit.read_text() == "old-socket"
    assert not list(package.parent.glob(".transaction.*"))
    assert not list(package.parent.glob(".previous.*"))


@pytest.mark.parametrize(
    "query_failure",
    ["masked", "query-transport-failure", "query-rc-mismatch"],
)
def test_invalid_systemd_query_fails_closed_before_install_writes(
    tmp_path: Path,
    query_failure: str,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, _, _ = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
    )
    if query_failure == "masked":
        (state / "socket.enabled").write_text("masked", encoding="ascii")
    else:
        (state / query_failure).touch()
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)

    result = _run_bash(INSTALLER, "--apply", env=_agent_test_env(fake_bin, install_root))

    assert result.returncode != 0
    assert (package / "old-agent.txt").is_file()
    assert service.read_text() == "old-service"
    assert socket_unit.read_text() == "old-socket"
    assert not (install_root / "var/lib/dgx-spark-ops-agent/jobs").exists()
    assert not list(package.parent.glob(".transaction.*"))


@pytest.mark.parametrize(
    ("termination", "expected_status"),
    [("exit 37", 37), ("kill -TERM $$", 143)],
)
def test_exit_and_term_restore_an_active_upgrade_transaction(
    tmp_path: Path,
    termination: str,
    expected_status: int,
) -> None:
    bash = _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX signal and metadata semantics")
    fake_bin, _, _, _ = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
    )
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)
    command = "; ".join(
        (
            f'source "{INSTALLER.as_posix()}"',
            "capture_systemd_snapshot",
            "begin_install_transaction",
            "quiesce_existing_agent",
            "snapshot_install_transaction",
            "install_package",
            termination,
        )
    )

    result = subprocess.run(
        [bash, "-c", command],
        cwd=ROOT,
        env={**os.environ, **_agent_test_env(fake_bin, install_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == expected_status
    assert (package / "old-agent.txt").is_file()
    assert service.read_text() == "old-service"
    assert socket_unit.read_text() == "old-socket"
    assert not list(package.parent.glob(".transaction.*"))


def test_bad_openssl_output_never_publishes_a_key(tmp_path: Path) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX ownership semantics")
    fake_bin, _, _, _ = _fake_agent_system(
        tmp_path,
        enabled=False,
        active=False,
        health_results=("pass",),
    )
    _write_executable(fake_bin / "openssl", "#!/bin/sh\nprintf 'not-a-valid-key\\n'\n")
    install_root = tmp_path / "root"

    result = _run_bash(INSTALLER, "--apply", env=_agent_test_env(fake_bin, install_root))

    key_dir = install_root / "etc/dgx-spark-manager"
    assert result.returncode != 0
    assert not (key_dir / "ops-agent.key").exists()
    assert not list(key_dir.glob(".ops-agent.key.*"))


def test_installer_validates_existing_key_and_updates_package_atomically() -> None:
    script = _read(INSTALLER)

    assert "-L \"$KEY_FILE\"" in script
    assert "validate_existing_key" in script
    assert "stat -c '%U:%G:%a'" in script
    assert "root:dgx-spark-ops:640" in script
    assert "mktemp -d" in script
    assert re.search(r"mv\s+--\s+\"\$[A-Z_]+\"\s+\"\$PACKAGE_DIR\"", script)
    assert "JOBS_DIR" in script
    assert "rm -rf -- \"$JOBS_DIR\"" not in script


def test_installer_retains_package_backup_until_health_succeeds() -> None:
    script = _read(INSTALLER)

    install = script.index("install_package")
    probe = script.index("probe_health", install)
    commit = script.index("commit_install_transaction", probe)
    assert install < probe < commit
    assert "rollback_install_transaction" in script
    assert "trap transaction_exit EXIT" in script
    moved = script.index('mv -- "$STAGED_PACKAGE" "$PACKAGE_DIR"')
    active = script.index("PACKAGE_UPDATE_ACTIVE=true", moved)
    staging_cleanup = script.index('rmdir -- "$PACKAGE_STAGING"', moved)
    assert moved < active < staging_cleanup


def test_installer_snapshots_and_restores_complete_upgrade_transaction() -> None:
    script = _read(INSTALLER)

    assert "snapshot_install_transaction" in script
    assert "snapshot_unit" in script
    assert "restore_unit_snapshot" in script
    assert "SOCKET_ENABLED_STATE" in script
    assert "SOCKET_ACTIVE_STATE" in script
    assert "SERVICE_ENABLED_STATE" in script
    assert "SERVICE_ACTIVE_STATE" in script
    snapshot = script.index("snapshot_install_transaction")
    stop = script.index("systemctl stop dgx-spark-ops-agent.socket", snapshot)
    assert snapshot < stop
    rollback = script.index("rollback_install_transaction")
    restore_unit = script.index("restore_unit_snapshot", rollback)
    reload_units = script.index("systemctl daemon-reload", restore_unit)
    restore_state = script.index("restore_systemd_state", reload_units)
    assert rollback < restore_unit < reload_units < restore_state
    assert "ROLLBACK_IN_PROGRESS" in script
    assert "local original_status" in script


def test_installer_fails_closed_on_systemd_state_and_quiesces_before_writes() -> None:
    script = _read(INSTALLER)

    assert "query_systemd_state" in script
    assert "validate_systemd_state" in script
    assert "enabled-runtime" in script
    assert "not-found" in script
    assert "quiesce_existing_agent" in script
    apply = script.index("apply_installation()")
    capture = script.index("capture_systemd_snapshot", apply)
    quiesce = script.index("quiesce_existing_agent", capture)
    artifact_snapshot = script.index("snapshot_install_transaction", quiesce)
    key = script.index("create_key_if_missing", artifact_snapshot)
    assert apply < capture < quiesce < artifact_snapshot < key
    assert "verify_systemd_state" in script
    assert "enable --now dgx-spark-ops-agent.socket" not in script
    assert "try-restart" not in script


def test_installer_uses_exit_and_signal_transaction_guards() -> None:
    script = _read(INSTALLER)

    assert "TRANSACTION_ACTIVE" in script
    assert "transaction_exit" in script
    assert "trap transaction_exit EXIT" in script
    assert "trap 'exit 129' HUP" in script
    assert "trap 'exit 130' INT" in script
    assert "trap 'exit 143' TERM" in script
    assert "clear_transaction_traps" in script


def test_new_key_is_validated_before_atomic_publish() -> None:
    script = _read(INSTALLER)

    openssl = script.index('openssl rand -hex 32 > "$KEY_TEMP"')
    validate = script.index('validate_key_content "$KEY_TEMP"', openssl)
    metadata = script.index('chmod 0640 -- "$KEY_TEMP"', validate)
    publish = script.index('mv -- "$KEY_TEMP" "$KEY_FILE"', metadata)
    assert openssl < validate < metadata < publish
    assert "KEY_CREATED_THIS_RUN" in script


def test_health_probe_authenticates_response_without_printing_key() -> None:
    script = _read(INSTALLER)

    assert "canonical_bytes" in script
    assert "sign_message" in script
    assert "request_id" in script
    assert "compare_digest" in script
    assert "protocol_version" in script
    assert "_RESPONSE_FIELDS" in script
    assert "set(response) != _RESPONSE_FIELDS" in script
    assert "print(secret" not in script
    assert "cat \"$KEY_FILE\"" not in script


def test_installer_embedded_python_is_syntactically_valid() -> None:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", _read(INSTALLER), flags=re.DOTALL)

    assert len(blocks) == 2
    for block in blocks:
        compile(block, os.fspath(INSTALLER), "exec")


def test_uninstaller_preserves_jobs_and_key_unless_explicitly_purged() -> None:
    script = _read(UNINSTALLER)

    assert "Preview only" in script
    assert "--apply" in script
    assert "--purge-key" in script
    assert "--purge-jobs" in script
    assert "JOBS_DIR=/var/lib/dgx-spark-ops-agent/jobs" in script
    assert "KEY_FILE=/etc/dgx-spark-manager/ops-agent.key" in script
    assert "Preserving job logs" in script
    assert "Preserving Agent key" in script
    assert "read -r" in script
    assert "DGX_OPS_AGENT_INSTALL_ROOT" in script
    assert "DGX_OPS_AGENT_TEST_EUID" in script
    assert "assert_trusted_path_chain" in script
    assert script.count("assert_trusted_path_chain") >= 4
    assert "manager.db" not in script
    assert "/models" not in script


def test_uninstaller_prevalidates_every_target_before_strict_systemd_shutdown() -> None:
    script = _read(UNINSTALLER)

    assert "validate_all_removal_targets" in script
    assert "strict_shutdown" in script
    validate = script.index("validate_all_removal_targets", script.index("main()"))
    shutdown = script.index("strict_shutdown", validate)
    first_remove = script.index("rm -", shutdown)
    assert validate < shutdown < first_remove
    for name in (
        "PACKAGE_DIR",
        "KEY_FILE",
        "JOBS_DIR",
        "SERVICE_UNIT",
        "SOCKET_UNIT",
        "SOCKET_PATH",
        "SOCKET_DIR",
    ):
        assert script.count(f'"${name}"') >= 2
    assert "query_systemd_state" in script
    assert "verify_systemd_state" in script
    assert "|| true" not in script


def test_uninstall_apply_preserves_key_and_jobs_with_fake_system(tmp_path: Path) -> None:
    _bash()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "systemctl",
        "\n".join(
            (
                "#!/bin/sh",
                "case \"$1\" in",
                "  is-enabled) printf 'disabled\\n'; exit 1 ;;",
                "  is-active) printf 'inactive\\n'; exit 3 ;;",
                "  *) exit 0 ;;",
                "esac",
                "",
            )
        ),
    )
    install_root = tmp_path / "root"
    package = install_root / "usr/local/lib/dgx-spark-ops-agent/dgx_ops_agent"
    key = install_root / "etc/dgx-spark-manager/ops-agent.key"
    job = install_root / "var/lib/dgx-spark-ops-agent/jobs/job.json"
    service = install_root / "etc/systemd/system/dgx-spark-ops-agent.service"
    socket_unit = install_root / "etc/systemd/system/dgx-spark-ops-agent.socket"
    for path in (package / "server.py", key, job, service, socket_unit):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")

    result = _run_bash(
        UNINSTALLER,
        "--apply",
        env={
            "PATH": _fake_path(fake_bin),
            "DGX_OPS_AGENT_TESTING": "1",
            "DGX_OPS_AGENT_TEST_EUID": "0",
            "DGX_OPS_AGENT_INSTALL_ROOT": _shell_path(install_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert not package.exists()
    assert not service.exists()
    assert not socket_unit.exists()
    assert key.read_text(encoding="utf-8") == "keep"
    assert job.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "failure",
    ["fail-stop", "sticky-stop", "fail-disable", "query-transport-failure"],
)
def test_uninstall_systemd_failure_preserves_every_artifact(
    tmp_path: Path,
    failure: str,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX metadata semantics")
    fake_bin, state, _, _ = _fake_agent_system(
        tmp_path,
        enabled=True,
        active=True,
        health_results=("pass",),
    )
    (state / failure).touch()
    install_root = tmp_path / "root"
    package, service, socket_unit = _seed_old_agent(install_root)
    jobs = install_root / "var/lib/dgx-spark-ops-agent/jobs"
    jobs.mkdir(parents=True)
    job = jobs / "job.json"
    job.write_text("keep-job", encoding="utf-8")
    key = install_root / "etc/dgx-spark-manager/ops-agent.key"

    result = _run_bash(
        UNINSTALLER,
        "--apply",
        env={
            "PATH": _fake_path(fake_bin),
            "DGX_OPS_AGENT_TESTING": "1",
            "DGX_OPS_AGENT_TEST_EUID": "0",
            "DGX_OPS_AGENT_INSTALL_ROOT": _shell_path(install_root),
        },
    )

    assert result.returncode != 0
    assert (package / "old-agent.txt").is_file()
    assert service.read_text() == "old-service"
    assert socket_unit.read_text() == "old-socket"
    assert key.is_file()
    assert job.read_text() == "keep-job"


@pytest.mark.parametrize("target", ["package", "key", "jobs", "unit", "run"])
def test_uninstall_rejects_symlinked_purge_ancestors(
    tmp_path: Path,
    target: str,
) -> None:
    _bash()
    if os.name == "nt":
        pytest.skip("requires native POSIX symlink semantics")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "systemctl", "#!/bin/sh\nexit 0\n")
    install_root = tmp_path / "root"
    install_root.mkdir()
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    marker = sentinel / "must-remain"
    marker.write_text("safe", encoding="utf-8")
    arguments = ["--apply"]
    confirmation = None

    if target == "package":
        ancestor = install_root / "usr/local/lib/dgx-spark-ops-agent"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(sentinel, target_is_directory=True)
        (sentinel / "dgx_ops_agent").mkdir()
    elif target == "key":
        ancestor = install_root / "etc/dgx-spark-manager"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(sentinel, target_is_directory=True)
        (sentinel / "ops-agent.key").write_text("secret", encoding="utf-8")
        arguments.append("--purge-key")
        confirmation = "PURGE OPS AGENT KEY\n"
    elif target == "jobs":
        ancestor = install_root / "var/lib/dgx-spark-ops-agent"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(sentinel, target_is_directory=True)
        (sentinel / "jobs").mkdir()
        arguments.append("--purge-jobs")
        confirmation = "PURGE OPS AGENT JOBS\n"
    elif target == "unit":
        ancestor = install_root / "etc/systemd"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(sentinel, target_is_directory=True)
        (sentinel / "system").mkdir()
        (sentinel / "system/dgx-spark-ops-agent.service").write_text(
            "unit",
            encoding="utf-8",
        )
    else:
        ancestor = install_root / "run/dgx-spark-manager"
        ancestor.parent.mkdir(parents=True)
        ancestor.symlink_to(sentinel, target_is_directory=True)
        (sentinel / "ops-agent.sock").write_text("socket", encoding="utf-8")

    result = _run_bash(
        UNINSTALLER,
        *arguments,
        env={
            "PATH": _fake_path(fake_bin),
            "DGX_OPS_AGENT_TESTING": "1",
            "DGX_OPS_AGENT_TEST_EUID": "0",
            "DGX_OPS_AGENT_INSTALL_ROOT": _shell_path(install_root),
        },
        input_text=confirmation,
    )

    assert result.returncode != 0
    assert "unsafe" in result.stderr.lower() or "symlink" in result.stderr.lower()
    assert marker.read_text(encoding="utf-8") == "safe"


def test_manager_installer_orders_agent_health_before_compose_and_upserts_env() -> None:
    script = _read(MANAGER_INSTALLER)

    agent = script.index("install-ops-agent.sh --apply")
    group = script.index("getent group dgx-spark-ops")
    compose = script.index("docker compose build")
    assert agent < group < compose
    assert "upsert_env" in script
    for name in (
        "PUID",
        "PGID",
        "DOCKER_GID",
        "OPS_AGENT_GID",
        "HF_HOME_HOST",
        "MODEL_HOME_HOST",
    ):
        assert f'upsert_env "$ENV_FILE" "{name}"' in script
    assert "chmod 0600" in script


def test_env_upsert_is_idempotent_and_preserves_secrets(tmp_path: Path) -> None:
    bash = _bash()
    env_file = tmp_path / ".env"
    original_secret = "do-not-change-this-secret"
    env_file.write_text(
        "\n".join(
            (
                f"DGX_SECRET_KEY={original_secret}",
                "DGX_ADMIN_PASSWORD=existing-password",
                "PUID=7",
                "OPS_AGENT_GID=8",
                "CUSTOM_VALUE=keep-me",
                "",
            )
        ),
        encoding="utf-8",
    )
    command = (
        f'source "{MANAGER_INSTALLER.as_posix()}"; '
        f'upsert_env "{env_file.as_posix()}" PUID 1000; '
        f'upsert_env "{env_file.as_posix()}" OPS_AGENT_GID 1234; '
        f'upsert_env "{env_file.as_posix()}" PUID 1000'
    )

    result = subprocess.run(
        [bash, "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    values = dict(
        line.split("=", maxsplit=1)
        for line in env_file.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert values["DGX_SECRET_KEY"] == original_secret
    assert values["DGX_ADMIN_PASSWORD"] == "existing-password"
    assert values["CUSTOM_VALUE"] == "keep-me"
    assert values["PUID"] == "1000"
    assert values["OPS_AGENT_GID"] == "1234"
    assert sum(line.startswith("PUID=") for line in env_file.read_text().splitlines()) == 1


def test_env_upsert_preserves_literal_backslashes_without_injecting_lines(
    tmp_path: Path,
) -> None:
    bash = _bash()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DGX_SECRET_KEY=keep-secret\nCUSTOM_VALUE=keep-me\n",
        encoding="utf-8",
    )
    command = (
        f'source "{MANAGER_INSTALLER.as_posix()}"; '
        f'upsert_env "{_shell_path(env_file)}" MODEL_HOME_HOST '
        "'/safe\\nINJECTED=yes'"
    )

    result = subprocess.run(
        [bash, "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert "MODEL_HOME_HOST=/safe\\nINJECTED=yes" in lines
    assert "INJECTED=yes" not in lines
    assert "DGX_SECRET_KEY=keep-secret" in lines
    assert "CUSTOM_VALUE=keep-me" in lines


def test_env_example_declares_non_secret_agent_configuration() -> None:
    values = dict(
        line.split("=", maxsplit=1)
        for line in _read(ROOT / ".env.example").splitlines()
        if line and not line.startswith("#")
    )

    assert values["OPS_AGENT_GID"].isdigit()
    assert values["DGX_OPS_AGENT_SOCKET"] == "/run/dgx-spark-manager/ops-agent.sock"
    assert values["DGX_OPS_AGENT_KEY_FILE"] == "/run/secrets/ops-agent.key"
    assert int(values["DGX_OPS_AGENT_CONNECT_TIMEOUT_SECONDS"]) >= 1
    assert int(values["DGX_OPS_AGENT_READ_TIMEOUT_SECONDS"]) >= 22
    assert int(values["DGX_OPS_AGENT_OUTPUT_LIMIT_BYTES"]) >= 10_000
