# MVP baseline

Дата фиксации: 2026-05-03

## Контекст

Корпус: `project_local`
Eval file: `eval/queries.mvp.local.yaml`
Текущая основная collection: `bitrix_code_mvp_sparse`

## Результаты

```text
dense:
Summary: total=40, hit@5=16/40 (40%), hit@10=17/40 (42%)
Path-only Summary: total=40, path_hit@5=18/40 (45%), path_hit@10=18/40 (45%)

qdrant-sparse:
Summary: total=40, hit@5=29/40 (72%), hit@10=32/40 (80%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=32/40 (80%)

qdrant-hybrid:
Summary: total=40, hit@5=29/40 (72%), hit@10=34/40 (85%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=35/40 (88%)
