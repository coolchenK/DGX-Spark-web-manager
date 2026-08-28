from app.services.model_capabilities import (
    infer_model_capabilities,
    input_modalities,
    merge_runtime_model_capabilities,
    runtime_multimodal_parameters,
)


def test_qwen_conditional_generation_detects_image_and_video_inputs(tmp_path):
    (tmp_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "vision_config": {"model_type": "qwen3_5_vision"},
          "image_token_id": 248056,
          "video_token_id": 248057
        }
        """,
        encoding="utf-8",
    )
    (tmp_path / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")

    assert infer_model_capabilities(tmp_path) == [
        "chat",
        "completion",
        "image",
        "video",
    ]


def test_structured_model_card_can_declare_vision_for_gguf(tmp_path):
    (tmp_path / "README.md").write_text(
        """---
pipeline_tag: image-text-to-text
tags:
  - vision-language
---
# Vision model
""",
        encoding="utf-8",
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"projector")

    assert infer_model_capabilities(tmp_path) == ["chat", "completion", "image"]


def test_unstructured_card_mentions_do_not_create_false_multimodal_capabilities(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "This text-only model was evaluated beside image and video systems.",
        encoding="utf-8",
    )

    assert infer_model_capabilities(tmp_path) == ["chat", "completion"]


def test_speculative_draft_model_ignores_text_generation_card_tag(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["Qwen3DSparkModel"]}', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        """---
pipeline_tag: text-generation
tags:
  - speculative-decoding
---
# DSpark draft
""",
        encoding="utf-8",
    )

    assert infer_model_capabilities(tmp_path) == []


def test_runtime_capability_merge_and_catalog_metadata():
    capabilities = merge_runtime_model_capabilities(
        ["chat", "completion"],
        ["chat", "completion", "image", "video"],
    )

    assert capabilities == ["chat", "completion", "image", "video"]
    assert input_modalities(capabilities) == ["text", "image", "video"]
    assert runtime_multimodal_parameters("vllm") == [
        "mm_processor_kwargs",
        "media_io_kwargs",
    ]
    assert runtime_multimodal_parameters("sglang") == [
        "images_config",
        "use_audio_in_video",
    ]
