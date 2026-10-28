"""仓储层与配置：产物缺失时的错误语义、配置校验。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mooc_dropout.config import Settings
from mooc_dropout.errors import ModelNotReadyError
from mooc_dropout.repository.artifact_repository import ArtifactRepository


def test_missing_artifacts_raise_model_not_ready(tmp_path):
    repo = ArtifactRepository(tmp_path / "nope")
    assert repo.is_available is False
    with pytest.raises(ModelNotReadyError) as excinfo:
        repo.load()
    assert "model" in excinfo.value.details["missing"]


def test_repository_loads_artifacts(trained):
    settings, _ = trained
    repo = ArtifactRepository(settings.model_dir)
    assert repo.is_available is True
    artifacts = repo.load()

    assert artifacts.threshold > 0
    assert artifacts.version != "unknown"
    assert len(artifacts.builder.feature_names) > 40
    # 二次加载走缓存，返回同一对象
    assert repo.load() is artifacts


def test_invalid_log_level_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


def test_prod_rejects_wildcard_cors():
    with pytest.raises(ValidationError):
        Settings(env="prod", cors_origins="*")


def test_cors_origin_list_splits():
    settings = Settings(cors_origins="http://a.com, http://b.com")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_relative_paths_resolve_under_project_root():
    settings = Settings(data_dir="data")
    assert settings.resolved_data_dir.is_absolute()
    assert settings.resolved_data_dir.name == "data"
