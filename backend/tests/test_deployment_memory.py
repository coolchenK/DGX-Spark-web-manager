from types import SimpleNamespace

from app.services.deployment_memory import (
    DeploymentMemoryService,
    container_process_ids,
    parse_nvidia_compute_memory,
)


class FakeContainer:
    def __init__(self, pids, memory=0):
        self.pids = pids
        self.memory = memory

    def top(self, **_kwargs):
        return {"Titles": ["UID", "PID"], "Processes": [["1000", str(pid)] for pid in self.pids]}

    def stats(self, **_kwargs):
        return {"memory_stats": {"usage": self.memory}}


class FakeContainers:
    def __init__(self, values):
        self.values = values

    def get(self, reference):
        return self.values[reference]


def successful_runner(output):
    def run(*_args, **_kwargs):
        return SimpleNamespace(stdout=output)

    return run


def deployment(deployment_id, *, status="running", container_id="container-1"):
    return SimpleNamespace(
        id=deployment_id,
        status=status,
        container_id=container_id,
        container_name="deployment",
    )


def test_parse_nvidia_compute_memory_ignores_invalid_rows_and_sums_duplicate_pids():
    parsed = parse_nvidia_compute_memory("101, 4096\ninvalid, N/A\n101, 128\n102, 2.5\n")

    assert parsed == {101: 4224 * 1024 * 1024, 102: int(2.5 * 1024 * 1024)}


def test_container_process_ids_uses_pid_column():
    assert container_process_ids(FakeContainer([101, 202])) == {101, 202}


def test_snapshot_attributes_gpu_process_memory_and_marks_stopped_instances():
    docker_client = SimpleNamespace(
        containers=FakeContainers({"container-1": FakeContainer([101, 202], memory=1234)})
    )
    service = DeploymentMemoryService(
        docker_client,
        command_runner=successful_runner("101, 4096\n202, 512\n999, 2048\n"),
    )

    result = service.snapshot(
        [deployment("running"), deployment("stopped", status="exited", container_id="other")]
    )

    assert result["running"]["memory_used_bytes"] == 4608 * 1024 * 1024
    assert result["running"]["memory_source"] == "nvidia_smi"
    assert result["stopped"]["memory_used_bytes"] == 0
    assert result["stopped"]["memory_source"] == "stopped"


def test_snapshot_falls_back_to_container_memory_without_gpu_process():
    docker_client = SimpleNamespace(
        containers=FakeContainers({"container-1": FakeContainer([303], memory=987654)})
    )
    service = DeploymentMemoryService(
        docker_client,
        command_runner=successful_runner("101, 4096\n"),
    )

    result = service.snapshot([deployment("running")])

    assert result["running"]["memory_used_bytes"] == 987654
    assert result["running"]["memory_source"] == "container"
