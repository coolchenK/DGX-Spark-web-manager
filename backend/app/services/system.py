import csv
import io
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

NVIDIA_QUERY = (
    "name,driver_version,temperature.gpu,power.draw,memory.used,utilization.gpu"
)


def _optional_float(value: str) -> float | None:
    normalized = value.strip().replace("[", "").replace("]", "")
    if normalized.lower() in {"n/a", "not supported", ""}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_nvidia_smi(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 6:
            continue
        memory_mib = _optional_float(row[4])
        gpus.append(
            {
                "name": row[0].strip(),
                "driver_version": row[1].strip(),
                "temperature_c": _optional_float(row[2]),
                "power_w": _optional_float(row[3]),
                "memory_used_bytes": (
                    int(memory_mib * 1024 * 1024) if memory_mib is not None else None
                ),
                "utilization_percent": _optional_float(row[5]),
            }
        )
    return gpus


def read_os_name(os_release_path: Path) -> str | None:
    try:
        lines = os_release_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values.get("PRETTY_NAME") or values.get("NAME")


class SystemService:
    def __init__(self, disk_path: Path, os_release_path: Path | None = None):
        self.disk_path = disk_path
        self.os_release_path = os_release_path

    def _gpu_snapshot(self) -> list[dict[str, Any]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={NVIDIA_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        return parse_nvidia_smi(result.stdout)

    def snapshot(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        disk_target = self.disk_path if self.disk_path.exists() else Path(os.getcwd())
        disk = psutil.disk_usage(str(disk_target))
        boot_time = psutil.boot_time()
        host_os = read_os_name(self.os_release_path) if self.os_release_path else None
        return {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "os": host_os or f"{platform.system()} {platform.release()}",
            "kernel": platform.version(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(logical=True) or 0,
            },
            "memory": {
                "total_bytes": memory.total,
                "used_bytes": memory.used,
                "available_bytes": memory.available,
            },
            "disk": {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
            },
            "gpus": self._gpu_snapshot(),
            "uptime_seconds": max(0, int(time.time() - boot_time)),
        }
