"""Single-query execution against the authoritative V5 contract-v2 pipeline."""
from functools import lru_cache
import json

from backend.config import ROOT
from backend.adapters.artifacts import select
from backend.utils.paths import analysis_directory, output_path

STAGES = ('query','corpus','index','retrieval','provenance','reranking','context','reasoning','fusion','decision')


@lru_cache(maxsize=1)
def resources():
    from src.rag import build_evidence_index_v2 as index
    from src.rag import run_rag_retrieval_v3 as retrieval
    from src.rag import run_rag_retrieval as retrieval_v2
    from src.rag import build_semantic_association_v5 as method
    from src.rag.final_test.preflight import evidence as verified_evidence
    from src.rag.final_test.contracts import DEVELOPMENT_ID
    policy = index.v1.load_token_policy(ROOT)
    evidence = retrieval.load_index(ROOT, policy, DEVELOPMENT_ID)
    registry, chunks, verified = verified_evidence(ROOT)
    verified = {key:dict(value, chunk_text=chunks[key]['chunk_text']) for key,value in verified.items()}
    encoder = retrieval_v2.load_query_encoder(ROOT, policy)
    claims = method.load_decision_claims(ROOT, DEVELOPMENT_ID)
    return policy, evidence, verified, encoder, registry, claims


def query_record(session, flow_id):
    from src.rag.run_rag_retrieval import semantic_profile, QUERY_FIELDS
    if session['input']['mode'] == 'external_inference':
        from backend.adapters.external_runtime import record
        return dict(record(session, flow_id), evaluation_scope='external_runtime')
    from backend.adapters.phase1_adapter import detail
    flow = detail(session, flow_id)
    if flow['source_split'] != 'validation':
        raise ValueError('Fresh desktop execution is enabled for external and validation flows. Use historical review for held-out test results.')
    prefix = 'ustc' if session['input']['dataset']=='USTC-TFC2016' else 'cic'
    row = select(f'results/rag/interface/{prefix}_validation_interface.csv','query_id',flow_id)[0]
    row.update(semantic_profile(row['source_dataset'],row['predicted_class_label'],row['traffic_representation_json']))
    return {k:v for k,v in row.items() if k in QUERY_FIELDS}


def deterministic(query, stage, result):
    from src.rag import run_rag_retrieval_v3 as retrieval
    from src.rag import run_rag_retrieval as retrieval_v2
    from src.rag import run_provenance_aware_reranking as reranking
    from src.rag.build_trusted_contexts import construct
    from src.rag.run_rag_retrieval import config_table
    from src.rag.final_test.contracts import RERANK_CONFIG, CONTEXT_CONFIG

    external = query['source_split'] == 'external_inference'
    stage('corpus','running','Validating the current V5 TRAIN-association evidence bundle')
    policy, evidence, verified, encoder, registry, claims = resources()
    chunks = {row['chunk_id']:row for row in evidence.chunks}
    stage('corpus','reused','Authoritative V5 evidence bundle and approved claim registry verified')
    stage('index','reused','Validated v5f_validation Index V2 loaded read-only; encoder cached')
    stage('retrieval','running','Running hypothesis-balanced TF-IDF, dense BGE and hybrid RRF retrieval')
    if external:
        from backend.external_phase2.v5_contract import kernel_query, retrieval_engine, externalize
        kernel = kernel_query(query)
        _, candidates = next(retrieval_engine().balanced_batch(evidence,[kernel],encoder,policy,
            source_split='external_inference'))
    else:
        kernel = query
        _, candidates = next(retrieval.balanced_batch(evidence,[query],encoder,policy,source_split='validation'))
    stage('retrieval','complete',f'{len(candidates)} fresh V5 candidates across three methods')
    result['provenance'] = [verified[key] for key in sorted({row['chunk_id'] for row in candidates})]
    stage('provenance','complete','Current V5 verifier checked trusted, authorized KEEP evidence')

    saved = config_table(ROOT,RERANK_CONFIG)
    config = dict(method=saved['method'], top_k=int(saved['top_k']), profile=saved['profile'],
        source_preferences=json.loads(saved['source_preferences']),
        **{key:float(saved[key]) for key in ('retrieval_weight','source_preference_weight',
            'task_alignment_weight','relationship_weight')})
    selected = [row for row in candidates if row['retrieval_method'] == config['method']]
    ranked = reranking.rerank(kernel,selected,verified,config,evidence=chunks,decision_claims=claims)
    stage('reranking','complete','Current task-relevance reranker applied approved-claim signals')

    context_cfg = config_table(ROOT,CONTEXT_CONFIG)
    limits = dict(top_n=int(context_cfg['top_n']), token_budget=int(context_cfg['token_budget']),
        max_per_document=int(context_cfg['max_chunks_per_document']),
        diversity_threshold=float(context_cfg['diversity_threshold']))
    if external:
        from backend.external_phase2.trusted_context_runtime import build_external_trusted_context
        from backend.external_phase2.v5_contract import externalize
        result['retrieval'] = [dict(externalize(row,query),document=chunks[row['chunk_id']]) for row in candidates]
        result['reranking'] = [externalize(row,query) for row in ranked]
        stage('context','running','Selecting external evidence-diverse V5 trusted context')
        result['context'] = build_external_trusted_context(query,result['reranking'],chunks,verified,policy,**limits)
        stage('context','complete','External V5 trusted context constructed; benchmark scopes unchanged')
    else:
        result['retrieval'] = [dict(row,document=chunks[row['chunk_id']]) for row in candidates]
        result['reranking'] = ranked
        stage('context','running','Constructing evidence-diverse V5 trusted context')
        result['context'] = construct(query,ranked,chunks,verified,policy,source_split='validation',
            profile='evidence_diverse',**limits)
        result['context']['evidence'] = json.loads(result['context']['evidence_records_json'])
        stage('context','complete','Current V5 context assembled with claim-first class-balanced exposure')


def reasoning(query, result, stage):
    from src.rag import association_response_v2 as contract
    from src.evaluation.finalize_v5_contract_v2_stage9 import fuse_accepted
    from backend.external_phase2 import llm_runtime
    _,_,verified,_,registry,_ = resources()
    context = result['context']
    payload = {key:context[key] for key in
        ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
    payload['trusted_evidence'] = json.loads(context['evidence_records_json'])
    state = contract.admissible(payload,registry,verified,source_split='validation')
    prompt = dict(query_id=query['query_id'],source_dataset=query['source_dataset'],payload=payload,
        prompt_text=contract.prompt(payload,registry,verified,state),
        response_schema=contract.schema(state,query['query_id']))
    status = llm_runtime.check_llm_runtime()
    if status['status'] != 'READY':
        raise ValueError(status['message'])
    count = llm_runtime.token_count(prompt)
    stage('reasoning','running','Calling the frozen V5 contract-v2 loopback runtime')
    response = llm_runtime.run_validation_constrained_llm(prompt,count,registry,verified)
    validated = response['validated_response']
    result['reasoning'] = response
    stage('reasoning','complete','Seven-field response and approved claim linkage validated')
    record = dict(query_id=query['query_id'],validation_status='PASS',response_contract_version='v2',
        validated_response={key:validated[key] for key in contract.runtime.validator.FIELDS},
        derived_validator_payload={key:validated[key] for key in contract.runtime.validator.FIELDS},
        raw_v2_model_response={key:response[key] for key in contract.FIELDS},
        validated_claim_linkage=validated['validated_claim_linkage'],
        model_explanation=response['explanation'],narrative_validation_status='NOT_SEMANTICALLY_VALIDATED')
    result['fusion'] = fuse_accepted(context,record,source_split='validation')
    stage('fusion','complete','Authoritative V5 retain-classifier fusion applied')
    result['decision'] = dict(result['fusion'],
        supporting_evidence_ids=json.loads(result['fusion']['cited_chunk_ids']))
    stage('decision','complete','Fresh V5 final decision available')


def execute(session, flow_id, stage):
    result = dict(flow_id=flow_id,mode='fresh_execution',sources={},unavailable=[],blockers={},
        query={},retrieval=[],provenance=[],reranking=[],context={},reasoning={},fusion={},decision={},
        phase2_contract='final_4300_v5_contract_v2')
    completed = set()
    def report(key,status,message):
        if status in {'complete','reused'}:
            completed.add(key)
        stage(key,status,message)
    try:
        report('query','running','Building a structured query with the current semantic-profile function')
        query = query_record(session,flow_id)
        result['query'] = query
        report('query','complete','Structured traffic query ready; source identity preserved')
        deterministic(query,report,result)
        if query['source_split']=='external_inference':
            from backend.external_phase2.orchestrator import run_external_phase2
            run_external_phase2(session,query,result,report)
        else:
            reasoning(query,result,report)
    except (ValueError,ImportError,OSError,RuntimeError,KeyError,TypeError) as exc:
        for key in STAGES:
            if key not in completed:
                result['unavailable'].append(key)
                result['blockers'][key] = str(exc)
                stage(key,'failed' if key=='context' else 'unavailable',str(exc))
        if result.get('query',{}).get('source_split')=='external_inference':
            result['runtime_status']=dict(trusted_context='FAILED',llm='WAITING',fusion='WAITING')
    directory = analysis_directory(session['analysis_id'])
    number = next(i for i,row in enumerate(session['flows']) if row['id']==flow_id)
    path = output_path(directory/f'phase2-{number}.json')
    result['sources']['runtime'] = str(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    session.setdefault('phase2',{})[flow_id] = result
    session['status'] = 'complete' if result['decision'] else 'partial'
    return result
