from pathlib import Path

import pytest
from app.tasks.huggingface import (
    cache_repository_path,
    serialize_card_data,
    validate_repository_id,
)
from huggingface_hub import ModelCardData


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
