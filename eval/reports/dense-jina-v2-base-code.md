# Dense experiment: jina-v2-base-code

Дата: 2026-05-03

## Config

Dense model: `jinaai/jina-embeddings-v2-base-code`
Qdrant collection: `bitrix_code_mvp_jina_v2_base_code`
Config dir: `.indexer/experiments/jina-v2-base-code/configs`

## Indexing

```text
files scanned: 573
files indexed: 573
chunks created: 3659
scanned_mb: 3.3

dense_embed: 1630.31s
qdrant_upsert: 7.81s
manifest_replace: 7.88s
total: 1700.70s
max RSS observed: 3723 MB
```

## Eval
```text
dense:
Summary: total=40, hit@5=15/40 (38%), hit@10=20/40 (50%)
Path-only Summary: total=40, path_hit@5=18/40 (45%), path_hit@10=22/40 (55%)

qdrant-sparse:
Summary: total=40, hit@5=29/40 (72%), hit@10=32/40 (80%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=32/40 (80%)

qdrant-hybrid:
Summary: total=40, hit@5=28/40 (70%), hit@10=31/40 (78%)
Path-only Summary: total=40, path_hit@5=30/40 (75%), path_hit@10=33/40 (82%)

```
