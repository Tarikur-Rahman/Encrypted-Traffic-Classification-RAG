"""Offline V5 TEST identity checks. Never reads labels or loads models."""
from .contracts import *
from src.rag import build_semantic_association_v5 as method
from src.rag import run_v5_contract_v2 as v5

INDEX_FILES = tuple('artifacts/rag/v5f_validation/index/'+n for n in
                   ('tfidf_vectorizer.joblib','tfidf_matrix.npz','dense_embeddings.npy','dense_chunk_ids.csv'))
POLICY_FILES = (RETRIEVAL_CONFIG,RERANK_CONFIG,CONTEXT_CONFIG,INDEX_CONFIG)


def frozen_inputs(root):
    bundle=json.loads(safe_path(root,BUNDLE).read_text(encoding='utf-8'))
    paths=[BUNDLE,HISTORICAL_MANIFEST,LEGACY_QUERIES,LEGACY_QUERY_RECEIPT,V5_RUN+'configuration.json',
           *POLICY_FILES,*INDEX_FILES,CHUNKS,DOCUMENTS,MANIFEST,QUALITY,CLAIMS]
    paths += [method.BASE+n for n in bundle['files']]
    saved=json.loads(safe_path(root,V5_RUN+'configuration.json').read_text(encoding='utf-8'))
    paths += list(saved['tokenizer_sha256'])
    paths += [p.relative_to(root).as_posix() for folder in ('src/rag','src/evaluation') for p in (root/folder).rglob('*.py')]
    return sorted(set(paths))


def evidence(root):
    # Existing frozen derivation/quality validators; no fitting, retrieval or inference.
    records=method.verify_bundle(root)
    require(method.load_decision_claims(root,DEVELOPMENT_ID)==method.derived_tables(records)[2],'Claim registry changed')
    registry={r['claim_id']:r for r in records if r['decision']=='APPROVED'}
    chunks=v5.runtime.index.validated_chunks(root,inputs=v5.runtime.index.input_paths(DEVELOPMENT_ID),
                                            outputs=v5.runtime.index.output_paths(DEVELOPMENT_ID))
    verified=v5.runtime.provenance.load_verified(root,DEVELOPMENT_ID)
    return registry,{r['chunk_id']:r for r in chunks},verified


def check_settings(root):
    require(sha(root,BUNDLE)==BUNDLE_SHA256,'Frozen TRAIN bundle differs')
    receipts=list(rows(root,LEGACY_QUERY_RECEIPT))
    require(len(receipts)==1 and receipts[0]['status']=='PASS','Frozen TEST observations not committed')
    identity=[r for r in json.loads(receipts[0]['output_metadata_json']) if r['path']==LEGACY_QUERIES]
    require(len(identity)==1 and snapshot(root,[LEGACY_QUERIES])==identity,'Historical TEST observations changed')
    saved=json.loads(safe_path(root,V5_RUN+'configuration.json').read_text(encoding='utf-8'))
    unsigned=dict(saved); signature=unsigned.pop('configuration_sha256')
    require(v5.runtime.digest(json_text(unsigned))==signature,'Frozen V5 configuration damaged')
    require(saved['code_sha256']['src/rag/positive_evidence_v5.py']==VALIDATION_VALIDATOR_SHA256,'Wrong validation validator parent')
    require(saved['response_contract_version']=='v2','Wrong V5 response contract')
    require(all(saved[k]==v for k,v in dict(model=v5.runtime.MODEL,endpoint=v5.runtime.ENDPOINT,
                max_model_len=FROZEN['max_model_len'],max_tokens=FROZEN['max_completion_tokens'],temperature=0,seed=42).items()),'Frozen V5 runtime differs')
    require(saved['token_audit']['reserved_completion_tokens']==saved['max_tokens']
            and saved['token_audit']['max_model_len']==saved['max_model_len'],'Frozen V5 token budget differs')
    for name,expected in saved['input_sha256'].items():
        require(sha(root,name)==expected,'Frozen validation dependency changed: '+name)
    for name,expected in saved['tokenizer_sha256'].items():
        require(sha(root,name)==expected,'Frozen tokenizer changed')
    # The original validation receipts remain untouched; TEST records the new
    # split-aware implementation hashes rather than rewriting old identities.
    for p in POLICY_FILES:
        values=config(root,p)
        for item in json.loads(values['input_metadata_json']):
            require(snapshot(root,[item['path']])==[item],'Frozen V5 upstream changed: '+item['path'])
    r,k,ctx=(config(root,p) for p in (RETRIEVAL_CONFIG,RERANK_CONFIG,CONTEXT_CONFIG))
    require(r['retrieval_profile']=='hypothesis_balanced','Wrong retrieval semantics')
    require(k['profile']=='task_relevance','Wrong reranking semantics')
    require(ctx['profile']=='evidence_diverse' and ctx['top_n']=='5' and ctx['token_budget']=='2048'
            and ctx['max_chunks_per_document']=='0' and float(ctx['diversity_threshold'])==1.0,'Wrong context semantics')
    require(config(root,INDEX_CONFIG)['development_id']==DEVELOPMENT_ID,'Wrong index')
    evidence(root)
    return saved


def check(root,*,scan_queries=True):
    root=root.resolve()
    check_settings(root)
    selected,excluded=cohort(root)
    if scan_queries:
        expected={r['query_id']:r for r in selected}; found=set()
        for row in rows(root,LEGACY_QUERIES):
            if row['query_id'] not in expected: continue
            query=dict(row,evaluation_scope=SCOPE)
            query_check(query)
            require(query['query_id'] not in found,'Duplicate selected source query')
            require(all(query[k]==expected[query['query_id']][k] for k in ('source_dataset','classifier_prediction','source_split')),'Cohort/query mismatch')
            found.add(query['query_id'])
        require(found==set(expected),'Missing frozen TEST queries')
    return dict(status='READY_FOR_4300_V5_TEST_PREFLIGHT',selected=len(selected),unique=len({r['query_id'] for r in selected}),
                excluded=len(excluded),duplicates=0,test_rows=TOTAL,validation_rows=0,
                prediction_strata={d+'/'+p:n for (d,p),n in TARGETS.items()},
                evaluation_scope=SCOPE,bundle_sha256=BUNDLE_SHA256,
                validation_validator_sha256=VALIDATION_VALIDATOR_SHA256,
                test_runtime_validator_sha256=sha(root,'src/rag/positive_evidence_v5.py'),
                ground_truth_used=0,classifier_confidence_used_as_evidence=0)
