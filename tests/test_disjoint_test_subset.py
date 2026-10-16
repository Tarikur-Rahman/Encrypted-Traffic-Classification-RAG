import pytest
from collections import Counter
from src.evaluation import prepare_disjoint_test_subset as module


def population():
    return [dict(query_id=f'{d}-TEST-{p}-{i:06d}',source_dataset=d,classifier_prediction=p,
        source_split='test',ground_truth='unused') for (d,p),n in module.TARGETS.items() for i in range(n+3)]


def test_counts_disjoint_deterministic_and_no_gt():
    rows=population()
    excluded={r['query_id'] for r in rows if r['query_id'].endswith('000000')}
    selected=module.select(rows,excluded)
    assert len(selected)==len({r['query_id'] for r in selected})==4319
    assert not excluded & {r['query_id'] for r in selected}
    assert Counter((r['source_dataset'],r['classifier_prediction']) for r in selected)==module.TARGETS
    assert selected==module.select([dict(r,ground_truth='changed') for r in reversed(rows)],excluded)
    assert all('ground_truth' not in r for r in selected)


def test_invalid_population_rejected():
    rows=population()
    with pytest.raises(ValueError,match='Non-TEST'):
        module.select([dict(rows[0],source_split='validation'),*rows[1:]],set())
    with pytest.raises(ValueError,match='Duplicate'):
        module.select([*rows,rows[0]],set())
    with pytest.raises(ValueError,match='Insufficient'):
        module.select(rows,{r['query_id'] for r in rows})


def test_existing_namespace_never_overwritten(tmp_path):
    (tmp_path/module.OUTPUT).mkdir(parents=True)
    with pytest.raises(ValueError,match='already exists'):
        module.prepare(tmp_path)
