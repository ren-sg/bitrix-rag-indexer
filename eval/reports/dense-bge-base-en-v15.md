# Dense experiment: bge-base-en-v15

Дата: 2026-05-03

## Config

Dense model: `BAAI/bge-base-en-v1.5`
Qdrant collection: `bitrix_code_mvp_bge_base_en_v15`
Config dir: `.indexer/experiments/bge-base-en-v15/configs`

## Indexing

```text
files scanned: 573
files indexed: 573
chunks created: 3659
scanned_mb: 3.3

dense_embed: 979.83s
qdrant_upsert: 8.09s
manifest_replace: 8.43s
total: 1033.27s
max RSS observed: 1345 MB
```

## Eval

```text
dense:
Summary: total=40, hit@5=15/40 (38%), hit@10=19/40 (48%)
Path-only Summary: total=40, path_hit@5=16/40 (40%), path_hit@10=20/40 (50%)

qdrant-sparse:
Summary: total=40, hit@5=29/40 (72%), hit@10=32/40 (80%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=32/40 (80%)

qdrant-hybrid:
Summary: total=40, hit@5=30/40 (75%), hit@10=34/40 (85%)
Path-only Summary: total=40, path_hit@5=31/40 (78%), path_hit@10=34/40 (85%)
```
