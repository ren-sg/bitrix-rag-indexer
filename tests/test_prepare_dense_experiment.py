from pathlib import Path

import yaml

from bitrix_rag_indexer.experiments.prepare import prepare_dense_experiment_config


def test_prepare_dense_experiment_config_updates_only_generated_configs(
    tmp_path: Path,
) -> None:
    base_config_dir = tmp_path / "configs"
    base_config_dir.mkdir()

    write_yaml(
        base_config_dir / "qdrant.yaml",
        {
            "url": "http://localhost:6333",
            "collection": "bitrix_code_mvp_sparse",
            "dense_vector_name": "dense",
            "sparse_vector_name": "sparse",
            "distance": "Cosine",
        },
    )
    write_yaml(
        base_config_dir / "embeddings.yaml",
        {
            "dense": {
                "provider": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "batch_size": 32,
                "cache_enabled": True,
                "cache_path": ".indexer/cache/embeddings.sqlite",
            },
            "sparse": {
                "enabled": True,
                "model": "Qdrant/bm25",
            },
        },
    )
    write_yaml(
        base_config_dir / "sources.mvp.yaml",
        {
            "sources": [],
        },
    )

    target_config_dir = prepare_dense_experiment_config(
        name="bge-m3-test",
        dense_model="BAAI/bge-m3",
        collection="bitrix_code_mvp_bge_m3",
        base_config_dir=base_config_dir,
        output_root=tmp_path / ".indexer" / "experiments",
    )

    generated_qdrant = read_yaml(target_config_dir / "qdrant.yaml")
    generated_embeddings = read_yaml(target_config_dir / "embeddings.yaml")

    assert generated_qdrant["collection"] == "bitrix_code_mvp_bge_m3"
    assert generated_embeddings["dense"]["model"] == "BAAI/bge-m3"

    original_qdrant = read_yaml(base_config_dir / "qdrant.yaml")
    original_embeddings = read_yaml(base_config_dir / "embeddings.yaml")

    assert original_qdrant["collection"] == "bitrix_code_mvp_sparse"
    assert original_embeddings["dense"]["model"] == "BAAI/bge-small-en-v1.5"

    assert (target_config_dir / "sources.mvp.yaml").exists()
    assert (target_config_dir.parent / "README.md").exists()


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    assert isinstance(data, dict)
    return data
