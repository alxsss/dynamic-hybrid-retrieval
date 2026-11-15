from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.initialize import initialize_repository
from dynamic_hybrid_retrieval.models import Document


def test_initialize_builds_complete_repository(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    source.write_text(
        Document(
            document_id="doc-1",
            title="Search",
            body="hybrid retrieval",
            updated_at=__import__("datetime").datetime(
                2026, 11, 14, tzinfo=__import__("datetime").UTC
            ),
            version=1,
        ).model_dump_json()
        + "\n"
    )
    manifest = initialize_repository(source, tmp_path / "repository", embedding_dimension=64)
    assert manifest.active_generation == "generation-000001"
    assert (tmp_path / "repository" / "active.json").is_file()


def test_initialize_rejects_empty_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    with pytest.raises(ValueError, match="at least one"):
        initialize_repository(source, tmp_path / "repository")
