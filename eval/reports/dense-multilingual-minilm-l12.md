# Dense experiment: multilingual-minilm-l12

Дата: 2026-05-03

## Config

Dense model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
Qdrant collection: `bitrix_code_mvp_multilingual_minilm_l12`
Config dir: `.indexer/experiments/multilingual-minilm-l12/configs`

## Indexing

```text
files scanned: 573
files indexed: 573
chunks created: 3659
scanned_mb: 3.3

dense_embed: 81.48s
qdrant_upsert: 7.00s
manifest_replace: 8.50s
total: 129.17s
max RSS observed: 1113 MB
```

## Eval

```text
dense:
Summary: total=40, hit@5=7/40 (18%), hit@10=7/40 (18%)
Path-only Summary: total=40, path_hit@5=10/40 (25%), path_hit@10=11/40 (28%)

qdrant-sparse:
Summary: total=40, hit@5=29/40 (72%), hit@10=32/40 (80%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=32/40 (80%)

qdrant-hybrid:
Summary: total=40, hit@5=27/40 (68%), hit@10=31/40 (78%)
Path-only Summary: total=40, path_hit@5=27/40 (68%), path_hit@10=31/40 (78%)
```
