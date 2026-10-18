import json

import pytest
from app.config import Settings
from app.models import ModelAsset
from app.runtime.base import ResolvedDeploymentSpec
from app.runtime.vllm import (
    AEON_MIXED_SPARK_0918_FIXED_IMAGE,
    AEON_MIXED_SPARK_0918_IMAGE,
    AEON_MIXED_SPARK_DFLASH_FIXED_IMAGE,
    AEON_MIXED_SPARK_IMAGE,
    AEON_MIXED_SPARK_OMNI_IMAGE,
    VllmAdapter,
)
from app.services.draft_models import classify_draft_candidate
from app.services.model_evidence import ModelEvidence


@pytest.mark.parametrize(
    "image, context_length, concurrency, batched_tokens, draft_tokens",
    [
        (AEON_MIXED_SPARK_IMAGE, 131072, 8, 8192, 7),
        (AEON_MIXED_SPARK_OMNI_IMAGE, 262144, 16, 16384, 10),
        (AEON_MIXED_SPARK_DFLASH_FIXED_IMAGE, 262144, 16, 16384, 10),
        (AEON_MIXED_SPARK_0918_IMAGE, 262144, 5, 16384, 9),
        (AEON_MIXED_SPARK_0918_FIXED_IMAGE, 262144, 5, 16384, 9),
        (AEON_MIXED_SPARK_0918_FIXED_IMAGE, 262144, 5, 16384, 7),
    ],
)
def test_aeon_spark_quality_recipe_preserves_model_card_contract(
    tmp_path, image, context_length, concurrency, batched_tokens, draft_tokens
):
    model = tmp_path / "aeon"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    patch = model / "modelopt-54367.py"
    patch.write_text("# model card patch\n")
    spec = ResolvedDeploymentSpec(
        name="AEON MIXED",
        model_path=str(model),
        api_model_name="aeon",
        runtime="vllm",
        image=image,
        context_length=context_length,
        memory_fraction=0.70,
        max_concurrency=concurrency,
        max_batched_tokens=batched_tokens,
        trust_remote_code=True,
        base_model_root=str(tmp_path),
        base_host_model_root="/host/cache",
        base_container_model_path="/models/aeon",
        draft_container_model_path="/models/dflash2",
        speculative_runtime_method="dflash",
        speculative={
            "method": "dflash", "draft_model_id": "draft", "num_speculative_tokens": draft_tokens
        },
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "medium"},
        generation_defaults={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.05,
        },
    )
    assert image in Settings.model_fields["allowed_vllm_images"].default.split(",")
    adapter = VllmAdapter(allowed_images={image}, model_roots=(tmp_path,))
    command = adapter.command(spec)
    assert command[:3] == ["vllm", "serve", "/models/aeon"]
    for flag, value in {
        "--max-model-len": str(context_length),
        "--gpu-memory-utilization": "0.7",
        "--max-num-seqs": str(concurrency),
        "--max-num-batched-tokens": str(batched_tokens),
        "--kv-cache-dtype": "fp8",
        "--attention-backend": "TRITON_ATTN",
        "--tool-call-parser": "qwen3_coder",
        "--reasoning-parser": "qwen3",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert {
        "--enable-chunked-prefill",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
    }.issubset(command)
    is_0918 = image in {AEON_MIXED_SPARK_0918_IMAGE, AEON_MIXED_SPARK_0918_FIXED_IMAGE}
    assert ("--enable-prefix-caching" in command) == is_0918
    assert ("--no-enable-prefix-caching" in command) != is_0918
    assert ("--enable-prompt-tokens-details" in command) == is_0918
    assert not {"--quantization", "--hf-overrides"}.intersection(command)
    expected_speculative = {
        "method": "dflash",
        "model": "/models/dflash2",
        "num_speculative_tokens": draft_tokens,
    }
    expected_environment = {"VLLM_USE_V2_MODEL_RUNNER": "1"}
    if image == AEON_MIXED_SPARK_IMAGE:
        assert "--compilation-config" not in command
    else:
        expected_environment.update(
            {"VLLM_ENABLE_CUDA_COMPATIBILITY": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
        )
        expected_speculative.update(
            {
                "attention_backend": "TRITON_ATTN",
                "num_speculative_tokens_per_batch_size": (
                    [[1, 2, draft_tokens], [3, 4, min(8, draft_tokens)],
                     [5, 8, 7], [9, 12, 6], [13, 16, 5]]
                    if is_0918 else [
                    [1, 1, 10], [2, 2, 10], [3, 4, 8], [5, 8, 7],
                    [9, 10, 6], [11, 12, 5], [13, 14, 4], [15, 16, 3],
                    ]
                ),
            }
        )
        assert json.loads(command[command.index("--compilation-config") + 1]) == {
            "cudagraph_mode": "FULL_AND_PIECEWISE"
        }
        for invalid_tokens in ((None, 8, 10) if is_0918 else (None, 7, 11)):
            error = (
                "supports 7 or 9" if is_0918 else f"requires num_speculative_tokens={draft_tokens}"
            )
            with pytest.raises(ValueError, match=error):
                adapter.command(spec.model_copy(update={
                    "speculative": spec.speculative.model_copy(
                        update={"num_speculative_tokens": invalid_tokens}
                    )
                }))
        with pytest.raises(ValueError, match="at most 16 concurrent"):
            adapter.command(spec.model_copy(update={"max_concurrency": 17}))
    assert json.loads(command[command.index("--speculative-config") + 1]) == expected_speculative
    assert json.loads(command[command.index("--limit-mm-per-prompt") + 1]) == {
        "image": 4,
        "video": 2,
    }
    assert json.loads(command[command.index("--override-generation-config") + 1]) == (
        spec.generation_defaults.model_dump(exclude_none=True)
    )
    assert adapter.environment(spec) == expected_environment
    assert adapter.entrypoint(spec) == []
    assert adapter.extra_volumes(spec) == ({} if is_0918 else {
        "/host/cache/aeon/modelopt-54367.py": {
            "bind": (
                "/usr/local/lib/python3.12/site-packages/vllm/"
                "model_executor/layers/quantization/modelopt.py"
            ),
            "mode": "ro",
        }
    })
    with pytest.raises(ValueError, match="auto-detect"):
        adapter.command(spec.model_copy(update={"quantization": "modelopt_fp4"}))
    patch.unlink()
    if is_0918:
        assert adapter.extra_volumes(spec) == {}
    else:
        with pytest.raises(ValueError, match="requires modelopt"):
            adapter.extra_volumes(spec)


def test_non_aeon_dflash_keeps_generic_runtime_settings(tmp_path):
    spec = ResolvedDeploymentSpec(
        name="Generic DFlash", model_path=str(tmp_path), api_model_name="generic",
        runtime="vllm", image="vllm:test", max_concurrency=32,
        draft_container_model_path="/models/draft", speculative_runtime_method="dflash",
        speculative={"method": "dflash", "draft_model_id": "draft", "num_speculative_tokens": 7},
    )
    adapter = VllmAdapter(allowed_images={spec.image}, model_roots=(tmp_path,))
    command = adapter.command(spec)
    assert command[:2] == ["--model", "/models"]
    assert json.loads(command[command.index("--speculative-config") + 1]) == {
        "method": "dflash", "model": "/models/draft", "num_speculative_tokens": 7,
    }
    assert not {"--attention-backend", "--compilation-config"}.intersection(command)
    assert adapter.environment(spec) == {}
    assert adapter.entrypoint(spec) is None
    assert adapter.extra_volumes(spec) == {}


@pytest.mark.parametrize(
    "draft_repository, expected",
    [
        ("z-lab/Qwen3.8-27B-DFlash2", "compatible"),
        ("other/Qwen3.8-27B-DFlash2", "incompatible"),
    ],
)
def test_aeon_accepts_only_author_documented_drafter(draft_repository, expected):
    target = ModelAsset(
        id="target",
        name="AEON",
        local_path="/models/aeon",
        repository_id="AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED",
        status="available",
        size_bytes=100,
    )
    draft = ModelAsset(
        id="draft",
        name="draft",
        local_path="/models/draft",
        repository_id=draft_repository,
        status="available",
        size_bytes=10,
    )
    common = dict(
        config={},
        generation_config={},
        tokenizer_fingerprint=None,
        card_text="",
        card_data={},
        card_deployment_values={},
        card_generation_values={},
        local_generation_values={},
        evidence_hash="a" * 64,
        warnings=[],
    )
    target_evidence = ModelEvidence(
        **common,
        model_path=target.local_path,
        target_model_ids=["AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16"],
        speculative_method=None,
    )
    draft_evidence = ModelEvidence(
        **common,
        model_path=draft.local_path,
        target_model_ids=["Qwen/Qwen3.8-27B"],
        speculative_method="dflash",
    )
    result = classify_draft_candidate(target, target_evidence, draft, draft_evidence, {"dflash"})
    assert result.status == expected
