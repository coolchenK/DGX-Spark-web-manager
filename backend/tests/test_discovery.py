import json
import os
from pathlib import Path

import pytest
from app.db import Database
from app.models import Deployment, ModelAsset
from app.services import discovery
from app.services.discovery import (
    DiscoveryService,
    container_candidate,
    hf_snapshot_is_complete,
    infer_runtime,
    parse_hf_cache_repository,
    resolve_hf_snapshot,
)
from sqlalchemy import select


def test_parse_hugging_face_cache_repository():
    assert (
        parse_hf_cache_repository(
            "models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
        )
        == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    )
    assert parse_hf_cache_repository(".locks") is None


def test_resolve_hf_snapshot_prefers_revision_reference(tmp_path):
    repository = tmp_path / "models--org--model"
    (repository / "refs").mkdir(parents=True)
    (repository / "snapshots" / "abc123").mkdir(parents=True)
    (repository / "refs" / "main").write_text("abc123\n", encoding="utf-8")

    assert resolve_hf_snapshot(repository) == repository / "snapshots" / "abc123"


def test_resolve_hf_snapshot_uses_newest_snapshot_without_main_ref(tmp_path):
    repository = tmp_path / "models--org--model"
    older = repository / "snapshots" / "older"
    newer = repository / "snapshots" / "newer"
    older.mkdir(parents=True)
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert resolve_hf_snapshot(repository) == newer


def test_resolve_hf_snapshot_returns_none_without_a_valid_snapshot(tmp_path):
    repository = tmp_path / "models--org--model"
    repository.mkdir()

    assert resolve_hf_snapshot(repository) is None


def test_hf_snapshot_requires_resolvable_repository_blobs(tmp_path):
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "commit123"
    blob = repository / "blobs" / ("a" * 64)
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    link = snapshot / "model.safetensors"
    try:
        link.symlink_to(Path("../../blobs") / blob.name)
    except OSError:
        pytest.skip("Symlink creation is unavailable")

    assert hf_snapshot_is_complete(repository, snapshot) is False

    blob.write_bytes(b"weights")

    assert hf_snapshot_is_complete(repository, snapshot) is True


def test_hf_snapshot_with_metadata_only_is_incomplete(tmp_path):
    repository = tmp_path / "models--org--metadata-only"
    snapshot = repository / "snapshots" / "commit123"
    blob = repository / "blobs" / ("c" * 64)
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    link = snapshot / "README.md"
    try:
        link.symlink_to(Path("../../blobs") / blob.name)
    except OSError:
        pytest.skip("Symlink creation is unavailable")

    blob.write_bytes(b"metadata")

    assert hf_snapshot_is_complete(repository, snapshot) is False


def test_scan_marks_broken_hf_snapshot_unavailable_and_recovers(settings):
    repository = settings.model_root_paths[0] / "models--org--broken-model"
    snapshot = repository / "snapshots" / "commit123"
    blob = repository / "blobs" / ("b" * 64)
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    link = snapshot / "model.safetensors"
    try:
        link.symlink_to(Path("../../blobs") / blob.name)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    database = Database(settings.database_url)
    database.create_schema()

    with database.session_factory() as db:
        first = DiscoveryService(settings.model_root_paths).scan_models(db)[0]
        assert first.status == "unavailable"
        assert first.local_path == str(repository)
        assert first.size_bytes == 0
        assert first.format is None

        blob.write_bytes(b"weights")
        recovered = DiscoveryService(settings.model_root_paths).scan_models(db)[0]

    assert recovered.status == "available"
    assert recovered.local_path == str(snapshot)
    assert recovered.size_bytes == len(b"weights")
    assert recovered.format == "safetensors"


def test_scan_hf_repository_without_snapshot_creates_unavailable_asset(settings):
    root = settings.model_root_paths[0]
    repository = root / "models--org--model"
    repository.mkdir(parents=True)
    database = Database(settings.database_url)
    database.create_schema()

    with database.session_factory() as db:
        discovered = DiscoveryService((root,)).scan_models(db)
        asset = db.scalar(select(ModelAsset).where(ModelAsset.repository_id == "org/model"))

    assert discovered == [asset]
    assert asset is not None
    assert asset.status == "unavailable"
    assert asset.local_path == str(repository)
    assert asset.commit_hash is None
    assert asset.revision is None
    assert asset.format is None
    assert asset.quantization is None
    assert asset.parameter_count is None
    assert asset.capabilities == []


def test_scan_existing_local_model_preserves_deleting_state_and_metadata(settings):
    root = settings.model_root_paths[0]
    model_path = root / "local-model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        '{"architectures":["LocalForCausalLM"]}', encoding="utf-8"
    )
    database = Database(settings.database_url)
    database.create_schema()
    metadata = {
        "keep": "value",
        "_delete_task_id": "task-1",
        "_delete_original_status": "available",
    }

    with database.session_factory() as db:
        asset = ModelAsset(
            name="local-model",
            source="local",
            local_path=str(model_path),
            status="deleting",
            metadata_json=metadata,
        )
        db.add(asset)
        db.commit()

        discovered = DiscoveryService((root,)).scan_models(db)
        db.refresh(asset)

    assert discovered == [asset]
    assert asset.status == "deleting"
    assert asset.metadata_json == metadata


@pytest.mark.parametrize("lifecycle_status", ["deleting", "delete_failed"])
def test_scan_hf_without_snapshot_preserves_lifecycle_state_and_metadata(
    settings, lifecycle_status
):
    root = settings.model_root_paths[0]
    repository = root / "models--org--model"
    repository.mkdir(parents=True)
    database = Database(settings.database_url)
    database.create_schema()
    metadata = {
        "keep": "value",
        "_delete_task_id": "task-1",
        "_delete_original_status": "available",
    }

    with database.session_factory() as db:
        asset = ModelAsset(
            name="org/model",
            source="huggingface",
            repository_id="org/model",
            local_path=str(repository / "missing-snapshot"),
            status=lifecycle_status,
            metadata_json=metadata,
        )
        db.add(asset)
        db.commit()

        discovered = DiscoveryService((root,)).scan_models(db)
        db.refresh(asset)

    assert discovered == [asset]
    assert asset.status == lifecycle_status
    assert asset.metadata_json == metadata


def test_scan_marks_missing_asset_unavailable_and_recovers_when_snapshot_returns(settings):
    root = settings.model_root_paths[0]
    root.mkdir()
    repository = root / "models--Qwen--Qwen2.5-0.5B-Instruct"
    missing_snapshot = repository / "snapshots" / "missing"
    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        asset = ModelAsset(
            name="Qwen/Qwen2.5-0.5B-Instruct",
            source="huggingface",
            repository_id="Qwen/Qwen2.5-0.5B-Instruct",
            local_path=str(missing_snapshot),
            status="available",
            commit_hash="missing",
            revision="main",
            format="safetensors",
            quantization="awq",
            parameter_count="0.5B",
            capabilities=["chat", "completion"],
            metadata_json={"deployable": True},
        )
        db.add(asset)
        db.commit()
        asset_id = asset.id

        assert DiscoveryService((root,)).scan_models(db) == []
        db.refresh(asset)
        assert asset.status == "unavailable"
        assert asset.commit_hash is None
        assert asset.revision is None
        assert asset.format is None
        assert asset.quantization is None
        assert asset.parameter_count is None
        assert asset.capabilities == []
        assert asset.metadata_json == {}

        snapshot = repository / "snapshots" / "commit123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            '{"architectures":["Qwen2ForCausalLM"]}', encoding="utf-8"
        )
        (snapshot / "model.safetensors").write_bytes(b"weights")

        recovered = DiscoveryService((root,)).scan_models(db)
        restored = db.get(ModelAsset, asset_id)

    assert recovered == [restored]
    assert restored is not None
    assert restored.status == "available"
    assert restored.local_path == str(snapshot)
    assert restored.commit_hash == "commit123"
    assert restored.revision == "commit123"
    assert restored.format == "safetensors"
    assert restored.capabilities == ["chat", "completion"]


def test_scan_does_not_mark_assets_unavailable_when_root_scan_fails(settings, tmp_path):
    root_path = tmp_path / "inaccessible-models"

    class InaccessibleRoot:
        def __fspath__(self):
            return str(root_path)

        def exists(self):
            return True

        def is_dir(self):
            return True

        def iterdir(self):
            raise PermissionError("not readable")

    database = Database(settings.database_url)
    database.create_schema()
    with database.session_factory() as db:
        asset = ModelAsset(
            name="protected",
            source="local",
            local_path=str(root_path / "protected"),
            status="available",
            format="safetensors",
            capabilities=["chat"],
        )
        db.add(asset)
        db.commit()

        assert DiscoveryService((InaccessibleRoot(),)).scan_models(db) == []
        db.refresh(asset)

    assert asset.status == "available"
    assert asset.format == "safetensors"
    assert asset.capabilities == ["chat"]


def test_scan_models_skips_inaccessible_roots(settings):
    class InaccessibleRoot:
        def exists(self):
            raise PermissionError("not readable")

    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    service = DiscoveryService((InaccessibleRoot(),))

    with database.session_factory() as db:
        assert service.scan_models(db) == []


def test_infer_runtime_from_image_and_command():
    assert (
        infer_runtime("sglang-inkling:specforge", ["python", "-m", "sglang.launch_server"])
        == "sglang"
    )
    assert infer_runtime("vllm/vllm-openai:v0.27.1", ["--model", "nvidia/model"]) == "vllm"
    assert (
        infer_runtime(
            "nvidia/cuda:12.9.0-devel-ubuntu24.04",
            ["/opt/llamacpp/llama-server", "--model", "/models/model.gguf"],
        )
        == "llama_cpp"
    )
    assert infer_runtime("redis:7", ["redis-server"]) is None


def test_container_candidate_extracts_openai_endpoint_and_model():
    attrs = {
        "Id": "abc123",
        "Name": "/qwen38-dspark",
        "Config": {
            "Image": "sglang-inkling:specforge",
            "Cmd": [
                "python3",
                "-m",
                "sglang.launch_server",
                "--served-model-name",
                "qwen3.8-27b",
                "--port",
                "8000",
            ],
            "Labels": {},
        },
        "State": {"Status": "running", "Health": {"Status": "healthy"}},
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8001"}]}
        },
    }

    candidate = container_candidate(attrs)

    assert candidate is not None
    assert candidate["name"] == "qwen38-dspark"
    assert candidate["runtime"] == "sglang"
    assert candidate["endpoint_url"] == "http://127.0.0.1:8001"
    assert candidate["api_model_name"] == "qwen3.8-27b"
    assert candidate["managed"] is False


def test_stopped_llama_container_uses_host_binding_and_api_alias():
    attrs = {
        "Id": "llama123",
        "Name": "/qwen-gguf",
        "Config": {
            "Image": "nvidia/cuda:12.9.0-devel-ubuntu24.04",
            "Cmd": [
                "/opt/llamacpp/llama-server",
                "--model",
                "/models/model.gguf",
                "--alias",
                "qwen35-9b-gguf",
                "--port",
                "8000",
            ],
            "Labels": {},
        },
        "HostConfig": {
            "PortBindings": {
                "8000/tcp": [{"HostIp": "", "HostPort": "8014"}]
            }
        },
        "State": {"Status": "created"},
        "NetworkSettings": {"Ports": {}},
    }

    candidate = container_candidate(attrs)

    assert candidate is not None
    assert candidate["runtime"] == "llama_cpp"
    assert candidate["endpoint_url"] == "http://127.0.0.1:8014"
    assert candidate["api_model_name"] == "qwen35-9b-gguf"
    assert candidate["health"] == "unknown"


def test_scan_does_not_probe_stopped_container_endpoint_reused_by_live_service(
    settings, monkeypatch
):
    class Container:
        attrs = {
            "Id": "stopped-container-id",
            "Name": "/nemotron-dspark",
            "Config": {
                "Image": "vllm/vllm-openai:v0.27.1",
                "Cmd": [
                    "serve",
                    "/model",
                    "--served-model-name",
                    "nemotron-3.5-lightning",
                    "--port",
                    "8000",
                ],
                "Labels": {},
            },
            "State": {"Status": "exited"},
            "NetworkSettings": {"Ports": {}},
            "HostConfig": {
                "PortBindings": {
                    "8000/tcp": [{"HostIp": "", "HostPort": "8011"}]
                }
            },
        }

        def reload(self):
            return None

    class Containers:
        @staticmethod
        def list(*, all):
            assert all is True
            return [Container()]

    class DockerClient:
        containers = Containers()

    service = DiscoveryService(settings.model_root_paths)
    monkeypatch.setattr(service, "_docker_client", lambda: DockerClient())

    def unexpected_probe(_candidate):
        pytest.fail("a stopped container endpoint must not be probed")

    monkeypatch.setattr(service, "_probe", unexpected_probe)
    database = Database(settings.database_url)
    database.create_schema()

    with database.session_factory() as db:
        stopped = Deployment(
            name="nemotron-dspark",
            runtime="vllm",
            container_id="stopped-container-id",
            container_name="nemotron-dspark",
            endpoint_url="http://127.0.0.1:8000",
            api_model_name="nemotron-3.5-lightning",
        )
        live = Deployment(
            name="qwen36-final",
            runtime="vllm",
            container_id="live-container-id",
            container_name="qwen36-final",
            endpoint_url="http://127.0.0.1:8000",
            api_model_name="qwen36-35b-heretic",
        )
        db.add_all([stopped, live])
        db.commit()

        discovered = service.scan_containers(db)
        db.refresh(stopped)

    assert discovered == [stopped]
    assert stopped.api_model_name == "nemotron-3.5-lightning"
    assert stopped.status == "exited"
    assert stopped.endpoint_url == "http://127.0.0.1:8011"
    assert stopped.port == 8011
    assert stopped.health == "unknown"


def test_scan_does_not_mutate_managed_identity_during_container_rebuild(
    settings, monkeypatch
):
    deployment_id = "7bc72a18-ec87-49a9-8ad6-606d432feef9"

    def attrs(container_id, name, status):
        return {
            "Id": container_id,
            "Name": f"/{name}",
            "Config": {
                "Image": "vllm/vllm-openai:latest",
                "Cmd": [
                    "serve",
                    "/model",
                    "--served-model-name",
                    "qwen36-35b-heretic",
                    "--port",
                    "8000",
                ],
                "Labels": {
                    "com.dgx-spark-manager.managed": "true",
                    "com.dgx-spark-manager.deployment-id": deployment_id,
                },
            },
            "State": {"Status": status},
            "NetworkSettings": {"Ports": {}},
            "HostConfig": {
                "PortBindings": {
                    "8000/tcp": [{"HostIp": "", "HostPort": "8000"}]
                }
            },
        }

    class Container:
        def __init__(self, value):
            self.attrs = value

        def reload(self):
            return None

    class Containers:
        @staticmethod
        def list(*, all):
            assert all is True
            return [
                Container(attrs("old-id", f"dgx-backup-{deployment_id}", "created")),
                Container(attrs("replacement-id", "qwen36-final", "running")),
            ]

    class DockerClient:
        containers = Containers()

    service = DiscoveryService(settings.model_root_paths)
    monkeypatch.setattr(service, "_docker_client", lambda: DockerClient())
    monkeypatch.setattr(service, "_probe", lambda _candidate: {"health": "healthy"})
    database = Database(settings.database_url)
    database.create_schema()

    with database.session_factory() as db:
        deployment = Deployment(
            id=deployment_id,
            name="Qwen3.6-35B-A3B-heretic",
            runtime="vllm",
            container_id="old-id",
            container_name="qwen36-final",
            endpoint_url="http://127.0.0.1:8000",
            api_model_name="qwen36-35b-heretic",
            status="starting",
            health="unknown",
            managed=True,
        )
        db.add(deployment)
        db.commit()

        discovered = service.scan_containers(db)
        db.refresh(deployment)

    assert discovered == [deployment]
    assert deployment.container_id == "old-id"
    assert deployment.container_name == "qwen36-final"
    assert deployment.status == "starting"
    assert deployment.health == "unknown"


def test_model_metadata_is_derived_from_snapshot_and_config(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "quantization_config": {"quant_method": "awq"},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    infer = getattr(discovery, "infer_model_metadata", lambda *_args, **_kwargs: {})

    assert infer(snapshot, "Qwen/Qwen2.5-0.5B-Instruct") == {
        "commit_hash": "abc123",
        "format": "safetensors",
        "quantization": "awq",
        "parameter_count": "0.5B",
        "capabilities": ["chat", "completion"],
    }


def test_model_metadata_detects_multimodal_conditional_generation(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3.8-27B" / "snapshots" / "commit123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5_vision"},
                "image_token_id": 248056,
                "video_token_id": 248057,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    metadata = discovery.infer_model_metadata(snapshot, "Qwen/Qwen3.8-27B")

    assert metadata["capabilities"] == ["chat", "completion", "image", "video"]


def test_deployment_capabilities_follow_model_and_saved_model_path(settings):
    root = settings.model_root_paths[0]
    model_path = root / "qwen38-multimodal"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5_vision"},
                "image_token_id": 248056,
                "video_token_id": 248057,
            }
        ),
        encoding="utf-8",
    )
    database = Database(settings.database_url)
    database.create_schema()
    service = DiscoveryService((root,))

    with database.session_factory() as db:
        deployment = Deployment(
            name="qwen38",
            runtime="vllm",
            endpoint_url="http://127.0.0.1:8001",
            api_model_name="qwen38",
            config={"spec": {"model_path": str(model_path)}},
        )
        db.add(deployment)
        db.commit()

        capabilities = service._deployment_capabilities(db, deployment)

    assert capabilities == ["chat", "completion", "image", "video"]


def test_scan_uses_commit_as_revision_when_no_named_ref_exists(settings):
    repository = settings.model_root_paths[0] / "models--org--model"
    snapshot = repository / "snapshots" / "commit123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"architectures": ["CausalLM"]}')
    (snapshot / "model.safetensors").write_bytes(b"weights")
    from app.db import Database

    database = Database(settings.database_url)
    database.create_schema()
    service = DiscoveryService(settings.model_root_paths)
    with database.session_factory() as db:
        model = service.scan_models(db)[0]

    assert model.commit_hash == "commit123"
    assert model.revision == "commit123"


def test_scan_preserves_local_model_discovery_behavior(settings):
    root = settings.model_root_paths[0]
    model_path = root / "local-model-7B"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        '{"architectures":["LocalForCausalLM"]}', encoding="utf-8"
    )
    (model_path / "model.safetensors").write_bytes(b"weights")
    database = Database(settings.database_url)
    database.create_schema()

    with database.session_factory() as db:
        discovered = DiscoveryService((root,)).scan_models(db)

    assert len(discovered) == 1
    model = discovered[0]
    assert model.source == "local"
    assert model.repository_id is None
    assert model.local_path == str(model_path)
    assert model.status == "available"
    assert model.format == "safetensors"
    assert model.parameter_count == "7B"
    assert model.capabilities == ["chat", "completion"]


def test_directory_size_does_not_double_count_linked_blobs(tmp_path):
    repository = tmp_path / "model"
    blob = repository / "blobs" / "hash"
    snapshot = repository / "snapshots" / "abc"
    blob.parent.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob.write_bytes(b"weights")
    os.link(blob, snapshot / "model.safetensors")

    assert discovery.directory_size(repository) == 7


def test_managed_discovery_preserves_saved_deployment_spec():
    merge = getattr(discovery, "merge_deployment_config", None)
    assert callable(merge)

    saved = {
        "spec": {"context_length": 4096, "memory_fraction": 0.25},
        "route_alias": "saved-route",
        "estimated_memory_bytes": 123,
    }
    observed = {
        "command": ["--max-model-len", "4096"],
        "model_path": "/models/qwen",
        "route_alias": "saved-route",
    }

    assert merge(saved, observed, managed=True) == {
        **saved,
        "command": observed["command"],
        "model_path": observed["model_path"],
    }
