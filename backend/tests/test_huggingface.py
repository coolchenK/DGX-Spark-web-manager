from pathlib import Path

import pytest
from app.tasks.huggingface import cache_repository_path, validate_repository_id


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
