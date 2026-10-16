"""Canonical TEST scope, streaming I/O, immutable inputs and stage receipts.

Identity uses exact configuration plus path/size/mtime metadata, never hashes.
Completed stages are reused; changed inputs or outputs block rather than retune.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from src.rag.phase2_common import require, json_text, CLASSES, validate_query

RUN_ID = 'final_4300_v5_contract_v2'
BASE = 'results/rag/test/'+RUN_ID+'/'
LEGACY_BASE = 'results/rag/test/'
LEGACY_QUERIES = LEGACY_BASE+'stage01_structured_queries/test_query_records.csv'
LEGACY_QUERY_RECEIPT = LEGACY_BASE+'stage01_structured_queries/test_query_validation_checklist.csv'
HISTORICAL_MANIFEST = LEGACY_BASE+'stage08_llm/final_disjoint_4319_v4/subset_manifest.csv'
HISTORICAL_MANIFEST_SHA256 = 'a0b58c5bc11875935ec301384f963aaa736a990e6220f1a58b42fdd8f292ffef'
SOURCE_COUNTS = {'USTC-TFC2016':98255, 'CIC-Darknet2020':31723}
COUNTS = {'USTC-TFC2016':1700, 'CIC-Darknet2020':2600}
TOTAL = 4300
SCOPE = SUBSET_SCOPE = 'FIXED_HISTORICAL_TEST_COMPARISON'
STRATEGY = 'frozen_seed42_4319_query_id_order_keep_first_50_Tor'
DEVELOPMENT_ID = 'v5f_validation'
BUNDLE = 'data/raw/rag/evidence_sources_v2/v5f_validation/bundle.json'
BUNDLE_SHA256 = 'db7eb4aac6363985ea0524be14422a7bda62936c9453616457f381af02c6b01e'
VALIDATION_VALIDATOR_SHA256 = '9b47110801b0bd9a6a689f4d419ff28d99a257429e8d0724653d722a6a78caed'
V5_RUN = 'results/rag/v5f_validation/stage08_comparison/v5f_historical4205_comparison_contract_v2/'
TARGETS = {(d,p): (50 if p=='Tor' else 850) for d,labels in CLASSES.items() for p in labels}
STAGES = ('Structured Security Query','Cybersecurity Evidence Corpus','Evidence Indexing','RAG Retrieval',
          'Provenance Verification','Provenance-Aware Evidence Reranking','Trusted Context Construction',
          'Constrained LLM Reasoning','Evidence-Supported Decision Fusion','Final Classification')
FOLDERS = ('stage01_structured_queries','stage02_corpus','stage03_index','stage04_retrieval','stage05_provenance',
           'stage06_reranking','stage07_trusted_contexts','stage08_llm','stage09_fusion','stage10_final')
PRIMARY = ('test_query_records.csv','corpus_reference_summary.csv','index_reference_summary.csv',
           'retrieval_candidates.csv','provenance_verified_candidates.csv','reranked_candidates.csv',
           'trusted_context_records.csv','test_llm_responses.jsonl','test_fusion_records.csv','test_final_classification_records.csv')
CONFIG = ('test_query_configuration.csv','corpus_configuration.csv','index_configuration.csv','retrieval_configuration.csv',
          'provenance_configuration.csv','reranking_configuration.csv','trusted_context_configuration.csv',
          'test_llm_configuration.csv','test_fusion_configuration.csv','test_final_configuration.csv')
CHECKS = ('test_query_validation_checklist.csv','corpus_validation_checklist.csv','index_validation_checklist.csv',
          'retrieval_validation_checklist.csv','provenance_validation_checklist.csv','reranking_validation_checklist.csv',
          'trusted_context_validation_checklist.csv','test_llm_execution_validation_checklist.csv',
          'test_fusion_validation_checklist.csv','test_final_validation_checklist.csv')
DEV = V5_RUN
RETRIEVAL_CONFIG = 'results/rag/v5f_validation/stage04_retrieval/experiment_configuration.csv'
RERANK_CONFIG = 'results/rag/v5f_validation/stage06_reranking/reranking_configuration.csv'
CONTEXT_CONFIG = 'results/rag/v5f_validation/stage07_trusted_contexts/trusted_context_configuration.csv'
INDEX_CONFIG = 'metadata/rag/v5f_validation/stage03_index/evidence_index_configuration.csv'
CHUNKS = 'data/processed/rag/v5f_validation/evidence_chunks.csv'
DOCUMENTS = 'data/processed/rag/v5f_validation/evidence_documents_v2.csv'
MANIFEST = 'metadata/rag/v5f_validation/stage02_corpus/evidence_source_manifest_v2.csv'
QUALITY = 'metadata/rag/v5f_validation/stage02_corpus/evidence_content_quality_review_v2.csv'
CLAIMS = 'metadata/rag/v5f_validation/stage02_corpus/decision_evidence_claims.csv'
GT_NAME = 'test_query_ground_truth_evaluation_only.csv'
GT_PATH = LEGACY_BASE+'stage01_structured_queries/'+GT_NAME
FORBIDDEN = {'trueclassid','trueclasslabel','classifiercorrect','groundtruth','targetid','targetbinary','correctness','correct','ytrue',
             'flowid','srcip','dstip','timestamp','srcport','dstport'}
FROZEN = dict(run_id=RUN_ID,cohort_size=TOTAL,evaluation_scope=SCOPE,sampling_seed=42,sampling_strategy=STRATEGY,
              evidence_method='train-semantic-association-v2',response_contract_version='v2',
              bundle_sha256=BUNDLE_SHA256,validation_validator_sha256=VALIDATION_VALIDATOR_SHA256,
              fusion_policy='retain_classifier',classifier_override='FORBIDDEN',
              model='provenance-qwen3-4b',max_model_len=5120,max_completion_tokens=1024,temperature=0,
              retrieval_profile='hypothesis_balanced',reranking_profile='task_relevance',
              context_profile='evidence_diverse',context_top_n=5,context_token_budget=2048,
              max_chunks_per_document=0,diversity_threshold=1.0)
FROZEN.update({f'ground_truth_used_stage{stage}':0 for stage in range(4,11)})
FROZEN.update(correctness_used_before_evaluation=0,classifier_confidence_used_as_evidence=0)


def sha(root,name):
    digest=hashlib.sha256()
    with safe_path(root,name).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def select_cohort(records):
    records=list(records)
    require(len(records)==4319 and len({r['query_id'] for r in records})==4319,'Historical cohort must contain 4319 unique IDs')
    require([r['query_id'] for r in records]==sorted(r['query_id'] for r in records),'Historical query-ID order differs')
    counts=Counter(); selected=[]; excluded=[]
    for ordinal,row in enumerate(records,1):
        require(row['source_split']=='test' and '-TEST-' in row['query_id'] and '-VAL-' not in row['query_id'],'Non-TEST historical ID')
        require(int(row['selection_ordinal'])==ordinal and int(row['sampling_seed'])==42,'Historical order/seed differs')
        pair=(row['source_dataset'],row['classifier_prediction'])
        require(pair in TARGETS,'Unknown historical prediction stratum')
        counts[pair]+=1
        # Outcome fields, if supplied by a caller, are never projected or used.
        safe={k:row[k] for k in ('query_id','source_dataset','classifier_prediction','source_split','stratum','selection_ordinal','sampling_seed')}
        (selected if counts[pair]<=TARGETS[pair] else excluded).append(safe)
    require(counts==Counter({k:69 if k[1]=='Tor' else 850 for k in TARGETS}),'Historical stratum counts differ')
    require(len(selected)==TOTAL and len(excluded)==19,'4300 selection failed')
    return selected,excluded


def cohort(root):
    require(sha(root,HISTORICAL_MANIFEST)==HISTORICAL_MANIFEST_SHA256,'Historical cohort hash differs')
    return select_cohort(rows(root,HISTORICAL_MANIFEST))


def rel(stage, name=None):
    return BASE+FOLDERS[stage-1]+'/'+(name or PRIMARY[stage-1])


def scope(stage):
    return 'split_independent_frozen_artifacts' if stage in (2,3) else SUBSET_SCOPE if stage >= 8 else SCOPE


def safe_path(root,name,write=False):
    root = root.resolve()
    p = root/name
    require(p.resolve() == p and p.is_relative_to(root), 'Linked/escaping path: '+str(name))
    require(not p.exists() or (p.is_file() and p.stat().st_nlink == 1), 'Aliased/non-file artifact: '+str(name))
    if write:
        require(p.is_relative_to(root/BASE), 'Writes outside canonical TEST root prohibited')
    else:
        require(p.is_file(), 'Missing artifact: '+str(name))
    return p


def rows(root,name,*,evaluation=False):
    require(evaluation or ('ground_truth' not in name and 'test_evaluation_' not in name), 'GT is evaluation-only')
    csv.field_size_limit(100_000_000)
    with safe_path(root,name).open(encoding='utf-8-sig',newline='') as stream:
        if name.endswith('.jsonl'):
            for line in stream:
                yield json.loads(line)
        else:
            reader = csv.DictReader(stream)
            require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),'Duplicate/empty header')
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()),'Malformed CSV: '+name)
                yield row


def config(root,name):
    data = list(rows(root,name))
    require(len({r['field'] for r in data}) == len(data),'Duplicate configuration keys')
    return {r['field']:r['value'] for r in data}


def snapshot(root,names):
    return [dict(path=n,size_bytes=(p:=safe_path(root,n)).stat().st_size,mtime_ns=p.stat().st_mtime_ns) for n in sorted(set(names))]


def assert_safe(value):
    """Inspect nested keys; quoted cybersecurity evidence text is not a GT field."""
    if isinstance(value,dict):
        for k,v in value.items():
            require(''.join(c for c in k.lower() if c.isalnum()) not in FORBIDDEN,'Forbidden RAG field: '+k)
            if isinstance(v,str) and v[:1] in ('{','['):
                try:
                    nested = json.loads(v)
                except ValueError:
                    nested = v
                assert_safe(nested)
            else:
                assert_safe(v)
    elif isinstance(value,(list,tuple)):
        for item in value:
            assert_safe(item)


def query_check(row):
    assert_safe(row)
    validate_query(row,source_split='test')
    prefix = 'USTC' if row['source_dataset'] == 'USTC-TFC2016' else 'CIC'
    index = int(row['query_id'].rsplit('-',1)[1])
    require(row['query_id'] == f'{prefix}-TEST-{index:06d}' and 0 <= index < SOURCE_COUNTS[row['source_dataset']], 'Invalid TEST row identity')
    require(row.get('evaluation_scope') == SCOPE,'Missing full TEST scope')
    require(row['classifier_prediction'] == row['predicted_class_label'],'Prediction alias differs')
    labels=CLASSES[row['source_dataset']]
    probabilities=json.loads(row['class_probabilities_json'])
    require(isinstance(probabilities,dict) and set(probabilities)==set(labels),'TEST requires labeled canonical probability vectors')
    predicted=labels[int(float(probabilities[labels[1]])>=.5)] if len(labels)==2 else max(labels,key=lambda k:float(probabilities[k]))
    require(predicted==row['classifier_prediction'],'Prediction differs from frozen threshold/argmax rule')
    require(abs(float(row['classifier_confidence'])-max(map(float,probabilities.values())))<=1e-10
            and float(row['classifier_confidence'])==float(row['confidence']),'Confidence/probability aliases differ')
    require(row.get('classifier_model','lightgbm')=='lightgbm','Only frozen selected LightGBM is eligible')
    if 'source_row_index' in row:
        require(int(row['source_row_index'])==index,'Source ordinal differs from TEST query identity')
    if 'traffic_provenance_json' in row:
        provenance=json.loads(row['traffic_provenance_json'])
        require(provenance['source_split']=='test' and provenance['source_dataset']==row['source_dataset']
                and int(provenance['source_row_index'])==index,'Non-TEST/mismatched traffic provenance')
        require(provenance['classifier_run_id']==row['classifier_run_id'] and provenance['classifier_model']=='lightgbm',
                'Classifier provenance differs')
    if 'classifier_probability_vector' in row:
        require(json.loads(row['classifier_probability_vector'])==probabilities,'Probability vector alias differs')


def queries(root):
    counts,seen = Counter(),set()
    for row in rows(root,rel(1)):
        query_check(row)
        require(row['query_id'] not in seen,'Duplicate TEST query')
        seen.add(row['query_id']); counts[row['source_dataset']] += 1
        yield row
    require(dict(counts) == COUNTS,f'TEST population differs: {dict(counts)}')


@contextmanager
def writer(root,name,header):
    target = safe_path(root,name,write=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.partial')
    safe_path(root,temporary.relative_to(root).as_posix(),write=True)
    try:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            out = csv.DictWriter(stream,fieldnames=header)
            out.writeheader()
            yield out
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)


def table(root,name,records,header=None):
    records = iter(records)
    first = next(records,None)
    require(first is not None or header is not None,'Empty output needs schema: '+name)
    with writer(root,name,header or tuple(first)) as out:
        if first is not None:
            out.writerow(first)
        out.writerows(records)


def configuration(root,name,values):
    table(root,name,({'field':k,'value':json_text(v) if isinstance(v,(list,dict)) else v} for k,v in sorted(values.items())),('field','value'))


def groups(root,name,source,fields,stage,*,evaluation=False):
    counts = Counter(tuple(str(row.get(k,'')) for k in fields) for row in rows(root,source,evaluation=evaluation))
    table(root,name,(dict(zip(fields,key),count=n,evaluation_scope=scope(stage)) for key,n in sorted(counts.items())),(*fields,'count','evaluation_scope'))


def committed(root,stage,*,required=False):
    path = root/rel(stage,CHECKS[stage-1])
    if not path.exists():
        require(not required,'Stage not complete: '+str(stage))
        return False
    receipt = list(rows(root,rel(stage,CHECKS[stage-1])))
    require(len(receipt)==1 and receipt[0]['status']=='PASS','Stage receipt is not PASS')
    require(receipt[0]['stage_number']==str(stage) and receipt[0]['evaluation_scope']==scope(stage),'Stage receipt identity/scope differs')
    require(stage in (2,3) or int(receipt[0]['processed_rows'])==TOTAL,'Stage receipt population differs')
    saved = json.loads(receipt[0]['output_metadata_json'])
    require({rel(stage),rel(stage,CONFIG[stage-1])}<={r['path'] for r in saved},'Receipt is missing canonical primary/configuration')
    require(snapshot(root,[r['path'] for r in saved]) == saved,'Canonical stage outputs changed: '+str(stage))
    cfg = config(root,rel(stage,CONFIG[stage-1]))
    sources = json.loads(cfg['input_metadata_json'])
    require(snapshot(root,[r['path'] for r in sources]) == sources,'Frozen stage inputs changed: '+str(stage))
    require(cfg['frozen_configuration_json'] == json_text(FROZEN),'Frozen scientific settings changed')
    require(json.loads(cfg['input_sha256_json'])=={p:sha(root,p) for p in json.loads(cfg['input_sha256_json'])},'Frozen input hash changed')
    require(cfg['test_runtime_validator_sha256']==sha(root,'src/rag/positive_evidence_v5.py'),'TEST validator changed')
    return True


def begin(root,stage,inputs):
    if committed(root,stage):
        return None
    before = snapshot(root,inputs)
    cfg = dict(evaluation_scope=scope(stage),population_test_queries=TOTAL,evaluated_llm_test_queries=TOTAL if stage>=8 else 0,
               sampling_seed=42,sampling_strategy=STRATEGY,
               frozen_configuration_json=json_text(FROZEN),input_metadata_json=json_text(before),
               input_sha256_json=json_text({p:sha(root,p) for p in inputs}),
               test_runtime_validator_sha256=sha(root,'src/rag/positive_evidence_v5.py'))
    name = rel(stage,CONFIG[stage-1])
    if (root/name).exists():
        require(config(root,name) == {k:str(v) for k,v in cfg.items()},'Partial TEST run configuration/input conflict; no retuning permitted')
    else:
        require(not (root/rel(stage)).exists(),'Existing primary output lacks frozen configuration')
        configuration(root,name,cfg)
    return before


def finish(root,stage,before,outputs,count):
    require(snapshot(root,[r['path'] for r in before]) == before,'Inputs changed during stage')
    expected = TOTAL
    require(stage in (2,3) or count == expected,f'Stage {stage} incomplete population: {count}')
    names = list(set([*outputs,rel(stage,CONFIG[stage-1])]))
    table(root,rel(stage,CHECKS[stage-1]),[dict(stage_number=stage,stage_name=STAGES[stage-1],
          evaluation_scope=scope(stage),processed_rows=count,status='PASS',
          output_metadata_json=json_text(snapshot(root,names)))])


def require_stages(root,*stages):
    for stage in stages:
        committed(root,stage,required=True)


@contextmanager
def execution_lock(root):
    """One canonical writer; reuse the hardened cross-platform file lock."""
    from src.rag.run_constrained_llm_v3 import response_file
    target=safe_path(root,BASE+'pipeline.lock',write=True)
    with response_file(target,True):
        yield


def manifests(root,closure=False):
    result=[]
    for stage,title in enumerate(STAGES,1):
        passed = committed(root,stage)
        receipt = list(rows(root,rel(stage,CHECKS[stage-1])))[0] if passed else {}
        result.append(dict(stage_number=stage,stage_name=title,evaluation_scope=scope(stage),
                           population_rows=0 if stage in (2,3) else TOTAL,
                           processed_rows=receipt.get('processed_rows',0),canonical_primary_artifact=rel(stage),
                           plotting_artifacts=BASE+'plotting/',status='PASS' if passed else 'INCOMPLETE',
                           notes='Frozen split-independent references' if stage in (2,3) else 'FIXED_HISTORICAL_TEST_COMPARISON; not natural population weighted'))
    table(root,BASE+'phase2_test_stage_manifest.csv',result)
    paths=dict(cohort=rel(1,'cohort_manifest.csv'),excluded_ids=rel(1,'excluded_query_ids.csv'),
               query_records=rel(1),evaluation_only_labels=GT_PATH,
               stage8_raw=BASE+FOLDERS[7]+'/execution/raw_responses.jsonl',
               stage8_validated=BASE+FOLDERS[7]+'/execution/validated_responses.jsonl',
               stage8_rejected=rel(8,'test_llm_rejected_responses.jsonl'),
               stage8_errors=BASE+FOLDERS[7]+'/execution/transport_errors.jsonl',
               final_predictions=rel(10),metrics=rel(10,'test_evaluation_metrics.csv'),
               per_class=rel(10,'test_evaluation_per_class_metrics.csv'),
               confusion_matrices=rel(10,'test_evaluation_confusion_matrix.csv'),
               evidence_status=rel(10,'test_final_evidence_status_distribution.csv'),
               provenance=rel(10,'test_final_provenance_summary.csv'),figure_tables=BASE+'plotting/')
    configuration(root,BASE+'phase2_test_experiment_summary.csv',dict(FROZEN,output_paths=paths,
        test_runtime_validator_sha256=sha(root,'src/rag/positive_evidence_v5.py')))
    if closure:
        require(all(r['status']=='PASS' for r in result),'All ten validated stages required for closure')
        for name in (rel(10,'test_evaluation_validation_checklist.csv'),BASE+'plotting/plotting_validation_checklist.csv'):
            check=list(rows(root,name,evaluation=True))
            require(len(check)==1 and check[0]['status']=='PASS','Evaluation/plotting receipt must be PASS')
            for field in ('input_metadata_json','output_metadata_json'):
                old=json.loads(check[0][field])
                require(snapshot(root,[r['path'] for r in old])==old,'Evaluation/plotting closure inputs changed')
        table(root,BASE+'closure/phase2_final_test_closure.csv',result)
    return result


def validate_closure(root):
    """Read-only recheck of the published closure and post-hoc artifacts."""
    saved=list(rows(root,BASE+'closure/phase2_final_test_closure.csv'))
    require(len(saved)==10 and [int(r['stage_number']) for r in saved]==list(range(1,11)),'Incomplete/duplicate closure stages')
    for stage,row in enumerate(saved,1):
        committed(root,stage,required=True)
        require(row['stage_name']==STAGES[stage-1] and row['status']=='PASS' and row['evaluation_scope']==scope(stage),'Closure stage/scope changed')
        receipt=list(rows(root,rel(stage,CHECKS[stage-1])))[0]
        require(row['processed_rows']==receipt['processed_rows'],'Closure row count differs')
    for name in (rel(10,'test_evaluation_validation_checklist.csv'),BASE+'plotting/plotting_validation_checklist.csv'):
        check=list(rows(root,name,evaluation=True))
        require(len(check)==1 and check[0]['status']=='PASS','Final evaluation/plotting is not PASS')
        for field in ('input_metadata_json','output_metadata_json'):
            old=json.loads(check[0][field])
            require(snapshot(root,[r['path'] for r in old])==old,'Closure dependency changed')
    return True
