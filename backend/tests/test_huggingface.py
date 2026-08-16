from pathlib import Path
from types import SimpleNamespace

import pytest
from app.tasks import huggingface
from app.tasks.huggingface import (
    cache_repository_path,
    serialize_card_data,
    validate_repository_id,
)
from huggingface_hub import ModelCardData


def model_result(
    repository_id,
    *,
    tags=None,
    downloads=0,
    pipeline_tag=None,
):
    return SimpleNamespace(
        id=repository_id,
        downloads=downloads,
        likes=0,
        pipeline_tag=pipeline_tag,
        private=False,
        gated=False,
        last_modified=None,
        tags=tags or [],
    )


class FakeHuggingFaceApi:
    def __init__(self, models):
        self.models = models
        self.calls = []

    def list_models(self, **kwargs):
        self.calls.append(kwargs)
        return self.models[: kwargs["limit"]]


@pytest.mark.parametrize(
    "value",
    ["../model", "org/../../etc", "org/model/extra", "/absolute/model", "org\\model"],
)
def test_repository_validation_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        validate_repository_id(value)


def test_cache_repository_path_stays_inside_cache(tmp_path):
    result = cache_repository_path(tmp_path, "nvidia/Nemotron")

    assert result == Path(tmp_path, "models--nvidia--Nemotron")
    assert result.is_relative_to(tmp_path)


def test_model_card_data_uses_hub_serialization_contract():
    card = ModelCardData(license="apache-2.0", pipeline_tag="text-generation")

    assert serialize_card_data(card)["license"] == "apache-2.0"
    assert serialize_card_data({"license": "mit"}) == {"license": "mit"}
    assert serialize_card_data(None) == {}


def test_model_card_text_uses_download_contract_and_bounds_content(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hub"
    downloaded = cache_dir / "README.md"
    cache_dir.mkdir()
    downloaded.write_text("abcdef", encoding="utf-8")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.setattr(huggingface, "hf_hub_download", fake_download)
    service = huggingface.HuggingFaceService(cache_dir, token="hf_test_token")

    assert service.model_card_text("org/model", revision="abc123", max_chars=4) == "abcd"
    assert calls == [
        {
            "repo_id": "org/model",
            "filename": "README.md",
            "revision": "abc123",
            "cache_dir": cache_dir,
            "token": "hf_test_token",
        }
    ]


def test_model_card_text_rejects_downloaded_path_outside_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()
    outside = tmp_path / "README.md"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(huggingface, "hf_hub_download", lambda **_kwargs: outside)
    service = huggingface.HuggingFaceService(cache_dir)

    with pytest.raises(ValueError, match="safe regular file"):
        service.model_card_text("org/model")


def test_model_card_text_rejects_downloaded_directory(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()
    downloaded = cache_dir / "README.md"
    downloaded.mkdir()
    monkeypatch.setattr(huggingface, "hf_hub_download", lambda **_kwargs: downloaded)
    service = huggingface.HuggingFaceService(cache_dir)

    with pytest.raises(ValueError, match="safe regular file"):
        service.model_card_text("org/model")


@pytest.mark.parametrize("max_chars", [0, -1, 500_001])
def test_model_card_text_rejects_unbounded_limits(tmp_path, max_chars):
    service = huggingface.HuggingFaceService(tmp_path)

    with pytest.raises(ValueError, match="max_chars"):
        service.model_card_text("org/model", max_chars=max_chars)


def test_search_ranks_nvfp4_ahead_of_more_downloaded_model(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result("org/popular-model", downloads=1_000_000),
            model_result(
                "nvidia/model-NVFP4",
                tags=["compressed-tensors", "safetensors"],
                downloads=10,
                pipeline_tag="text-generation",
            ),
        ]
    )

    results = service.search("model", limit=2)

    assert [result["id"] for result in results] == [
        "nvidia/model-NVFP4",
        "org/popular-model",
    ]
    assert results[0]["spark_compatibility"] == {
        "level": "recommended",
        "score": 180,
        "reasons": ["NVFP4 量化", "压缩权重格式", "Safetensors 权重"],
    }


def test_search_preserves_hugging_face_order_for_equal_scores(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result("org/first", tags=["safetensors"], downloads=1),
            model_result("org/second", tags=["safetensors"], downloads=999),
        ]
    )

    results = service.search("model", limit=2)

    assert [result["id"] for result in results] == ["org/first", "org/second"]


def test_search_marks_gguf_only_model_for_review(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi([model_result("org/model-7B", tags=["gguf"])])

    result = service.search("model", limit=1)[0]

    assert result["spark_compatibility"] == {
        "level": "review",
        "score": -40,
        "reasons": ["需要额外运行时"],
    }


def test_search_scores_fifty_candidates_before_applying_safe_limit(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    candidates = [model_result(f"org/ordinary-{index}") for index in range(49)]
    candidates.append(model_result("org/best-NVFP4", tags=["safetensors"]))
    candidates.append(model_result("org/not-in-pool-NVFP4", tags=["compressed-tensors"]))
    api = FakeHuggingFaceApi(candidates)
    service.api = api

    results = service.search("model", limit=0)

    assert api.calls == [{"search": "model", "limit": 50, "full": True}]
    assert [result["id"] for result in results] == ["org/best-NVFP4"]


def test_search_does_not_recommend_oversized_nvfp4_model(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result(
                "org/model-27B-2.4T-NVFP4",
                tags=["compressed-tensors"],
            )
        ]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "review",
        "score": 30,
        "reasons": ["NVFP4 量化", "模型规模需评估", "压缩权重格式"],
    }


def test_search_ranks_compatible_model_before_higher_scoring_review_model(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result("org/review-NVFP4", tags=["vllm", "gguf"]),
            model_result("org/compatible-AWQ"),
        ]
    )

    results = service.search("model", limit=2)

    assert [result["id"] for result in results] == [
        "org/compatible-AWQ",
        "org/review-NVFP4",
    ]
    assert [result["spark_compatibility"]["score"] for result in results] == [30, 100]


@pytest.mark.parametrize("quantization", ["AWQ", "GPTQ"])
def test_search_scores_low_bit_quantization_signals(tmp_path, quantization):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [model_result(f"org/model-{quantization}", tags=[quantization.lower()])]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 30,
        "reasons": ["低比特量化"],
    }


def test_search_prefers_fp8_over_other_low_bit_signal(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [model_result("org/model-FP8-AWQ", tags=["fp8", "awq"])]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 35,
        "reasons": ["FP8 量化"],
    }


def test_search_puts_fp8_before_format_reasons_and_truncates_to_three(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result(
                "org/model",
                tags=["fp8", "compressed-tensors", "safetensors", "vllm"],
            )
        ]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 105,
        "reasons": ["FP8 量化", "压缩权重格式", "Safetensors 权重"],
    }


def test_search_only_counts_nvfp4_when_multiple_quantization_signals_exist(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [model_result("org/model-NVFP4-FP8-AWQ", tags=["nvfp4", "fp8", "awq"])]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 120,
        "reasons": ["NVFP4 量化"],
    }


@pytest.mark.parametrize("negative_prefix", ["not", "no", "non"])
def test_search_ignores_negated_and_inexact_compatibility_signals(tmp_path, negative_prefix):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result(
                f"org/{negative_prefix}-nvfp4-ready",
                tags=[
                    f"{negative_prefix}-fp8",
                    f"{negative_prefix}-awq",
                    f"{negative_prefix}-gptq",
                    "not-vllm-compatible",
                    "compressed-tensors-ready",
                    "not-safetensors",
                    "gguf-ready",
                ],
            )
        ]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {"level": "review", "score": 0, "reasons": []}


@pytest.mark.parametrize(
    ("repository_id", "tags"),
    [
        ("org/not_nvfp4", []),
        ("org/not.nvfp4", []),
        ("org/model", ["no_fp8"]),
        ("org/model", ["non.gptq"]),
    ],
)
def test_search_ignores_negated_quantization_with_non_alphanumeric_separator(
    tmp_path,
    repository_id,
    tags,
):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi([model_result(repository_id, tags=tags)])

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {"level": "review", "score": 0, "reasons": []}


@pytest.mark.parametrize(
    ("repository_id", "expected_score", "expected_reason"),
    [
        ("org/model-NVFP4", 120, "NVFP4 量化"),
        ("org/model_nvfp4", 120, "NVFP4 量化"),
        ("org/block-fp8", 35, "FP8 量化"),
        ("org/model.fp8", 35, "FP8 量化"),
        ("org/model-AWQ", 30, "低比特量化"),
    ],
)
def test_search_recognizes_bounded_quantization_tokens(
    tmp_path,
    repository_id,
    expected_score,
    expected_reason,
):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi([model_result(repository_id)])

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": expected_score,
        "reasons": [expected_reason],
    }


def test_search_forces_positive_scoring_gguf_only_model_to_review(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [model_result("org/model-NVFP4", tags=["vllm", "gguf"])]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "review",
        "score": 100,
        "reasons": ["NVFP4 量化", "需要额外运行时", "适配当前推理运行时"],
    }


def test_search_puts_gguf_only_downgrade_before_runtime_and_pipeline_reasons(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result(
                "org/model-NVFP4",
                tags=["vllm", "gguf"],
                pipeline_tag="text-generation",
            )
        ]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "review",
        "score": 110,
        "reasons": ["NVFP4 量化", "需要额外运行时", "适配当前推理运行时"],
    }


def test_search_treats_moe_total_parameter_count_as_capacity_risk(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [model_result("org/16x21B-NVFP4", tags=["compressed-tensors"])]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "review",
        "score": 30,
        "reasons": ["NVFP4 量化", "模型规模需评估", "压缩权重格式"],
    }


def test_search_ignores_parameter_size_embedded_in_alphanumeric_token(tmp_path):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi([model_result("org/modelx200B-NVFP4")])

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 120,
        "reasons": ["NVFP4 量化"],
    }


@pytest.mark.parametrize("runtime", ["vllm", "sglang"])
def test_search_scores_runtime_and_generation_task(tmp_path, runtime):
    service = huggingface.HuggingFaceService(tmp_path)
    service.api = FakeHuggingFaceApi(
        [
            model_result(
                "org/model",
                tags=[runtime],
                pipeline_tag="image-text-to-text",
            )
        ]
    )

    compatibility = service.search("model", limit=1)[0]["spark_compatibility"]

    assert compatibility == {
        "level": "compatible",
        "score": 30,
        "reasons": ["适配当前推理运行时", "生成任务"],
    }


def test_download_size_only_counts_selected_files():
    files = [
        {"name": "config.json", "size": 100},
        {"name": "model.safetensors", "size": 1_000},
        {"name": "README.md", "size": 50},
    ]
    calculate = getattr(huggingface, "selected_download_size", lambda *_args, **_kwargs: None)

    assert calculate(files, include=["*.json", "*.safetensors"], exclude=["README*"]) == 1_100


def test_disk_preflight_rejects_download_larger_than_free_space():
    validate = getattr(huggingface, "validate_disk_capacity", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="free disk space"):
        validate(total_bytes=10_000, existing_bytes=1_000, free_bytes=8_999)

    assert validate(total_bytes=10_000, existing_bytes=2_000, free_bytes=8_000) == 8_000


def test_download_handler_runs_disk_preflight_before_starting_cli(tmp_path, monkeypatch):
    service = huggingface.HuggingFaceService(tmp_path / "hub")
    monkeypatch.setattr(
        service,
        "info",
        lambda *_args: {
            "sha": "abc",
            "siblings": [{"name": "model.safetensors", "size": 10_000}],
            "total_size": 10_000,
        },
    )
    monkeypatch.setattr(huggingface.shutil, "which", lambda _name: "/usr/bin/hf")
    monkeypatch.setattr(
        huggingface.shutil,
        "disk_usage",
        lambda _path: huggingface.shutil._ntuple_diskusage(20_000, 11_001, 8_999),
    )
    monkeypatch.setattr(
        huggingface.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("CLI started before disk preflight"),
    )

    class Context:
        def update(self, **_kwargs):
            return None

    with pytest.raises(RuntimeError, match="free disk space"):
        service.download_handler(
            Context(),
            {"repository_id": "org/model", "revision": "main", "include": [], "exclude": []},
        )


def test_snapshot_integrity_check_requires_every_selected_file(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    files = [
        {"name": "config.json", "size": 2},
        {"name": "model.safetensors", "size": 100},
    ]
    verify = getattr(huggingface, "verify_snapshot_files", lambda *_args, **_kwargs: None)

    assert verify(snapshot, files, include=["config.json"], exclude=[]) == ["config.json"]
    with pytest.raises(RuntimeError, match="model.safetensors"):
        verify(snapshot, files, include=[], exclude=[])


def test_snapshot_integrity_allows_hub_links_to_repository_blobs(tmp_path):
    repository = tmp_path / "models--org--model"
    snapshot = repository / "snapshots" / "abc"
    blob = repository / "blobs" / "hash"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    is_safe = getattr(huggingface, "is_safe_snapshot_file", lambda *_args: False)

    assert is_safe(
        snapshot,
        snapshot / "model.safetensors",
        blob,
    ) is True
    assert is_safe(snapshot, snapshot / "../escape", tmp_path / "outside") is False


def test_huggingface_cli_uses_writable_cache_home(tmp_path):
    cache = tmp_path / "hf-cache" / "hub"
    build_environment = getattr(
        huggingface,
        "huggingface_environment",
        lambda *_args, **_kwargs: {},
    )

    environment = build_environment(cache, {"HOME": "/"})

    assert environment["HOME"] == str(cache.parent)
    assert environment["HF_HOME"] == str(cache.parent)
    assert environment["HF_HUB_CACHE"] == str(cache)
    assert environment["HF_XET_CACHE"] == str(cache.parent / "xet")
    assert environment["XDG_CACHE_HOME"] == str(cache.parent / ".cache")


def test_huggingface_cli_errors_are_sanitized():
    sanitize = getattr(huggingface, "sanitize_cli_output", lambda _value: "")

    result = sanitize("\x1b[31mPermission denied hf_abcdefghijklmnopqrstuvwxyz\x1b[0m")

    assert result == "Permission denied [REDACTED_HF_TOKEN]"
