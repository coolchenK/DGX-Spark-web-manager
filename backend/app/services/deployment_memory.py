from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from app.models import Deployment

MIB = 1024 * 1024
NVIDIA_COMPUTE_QUERY = (
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
)


def parse_nvidia_compute_memory(output: str) -> dict[int, int]:
    usage: dict[int, int] = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
            memory_mib = float(row[1].strip())
        except ValueError:
            continue
        if pid > 0 and memory_mib >= 0:
            usage[pid] = usage.get(pid, 0) + int(memory_mib * MIB)
    return usage


def container_process_ids(container: Any) -> set[int]:
    process_table = container.top(ps_args="-eo pid")
    titles = [str(value).strip().upper() for value in process_table.get("Titles", [])]
    try:
        pid_index = titles.index("PID")
    except ValueError:
        return set()
    pids: set[int] = set()
    for process in process_table.get("Processes", []):
        try:
            pid = int(process[pid_index])
        except (IndexError, TypeError, ValueError):
            continue
        if pid > 0:
            pids.add(pid)
    return pids


class DeploymentMemoryService:
    def __init__(self, docker_client: Any, *, command_runner=subprocess.run) -> None:
        self.docker_client = docker_client
        self.command_runner = command_runner

    def _nvidia_usage(self) -> tuple[dict[int, int], bool]:
        try:
            result = self.command_runner(
                ["nvidia-smi", *NVIDIA_COMPUTE_QUERY],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return {}, False
        return parse_nvidia_compute_memory(result.stdout), True

    @staticmethod
    def _container_memory(container: Any) -> int | None:
        try:
            stats = container.stats(stream=False)
            value = (stats.get("memory_stats") or {}).get("usage")
            return int(value) if isinstance(value, (int, float)) and value >= 0 else None
        except Exception:
            return None

    def snapshot(self, deployments: Iterable[Deployment]) -> dict[str, dict[str, Any]]:
        measured_at = datetime.now(UTC).isoformat()
        gpu_usage, nvidia_available = self._nvidia_usage()
        result: dict[str, dict[str, Any]] = {}
        for deployment in deployments:
            if deployment.status != "running":
                result[deployment.id] = {
                    "memory_used_bytes": 0,
                    "memory_source": "stopped",
                    "memory_measured_at": measured_at,
                }
                continue
            reference = deployment.container_id or deployment.container_name
            if not reference:
                result[deployment.id] = {
                    "memory_used_bytes": None,
                    "memory_source": "unavailable",
                    "memory_measured_at": measured_at,
                }
                continue
            try:
                container = self.docker_client.containers.get(reference)
                pids = container_process_ids(container)
            except Exception:
                container = None
                pids = set()
            gpu_bytes = sum(gpu_usage.get(pid, 0) for pid in pids)
            if gpu_bytes > 0:
                memory_used = gpu_bytes
                source = "nvidia_smi"
            elif container is not None:
                memory_used = self._container_memory(container)
                source = "container" if memory_used is not None else "unavailable"
            else:
                memory_used = None
                source = "unavailable"
            if nvidia_available and gpu_bytes == 0 and memory_used == 0:
                source = "container"
            result[deployment.id] = {
                "memory_used_bytes": memory_used,
                "memory_source": source,
                "memory_measured_at": measured_at,
            }
        return result
