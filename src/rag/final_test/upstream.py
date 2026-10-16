"""Full-population TEST adapters reusing frozen computational kernels."""
import json
import re
from collections import Counter
from dataclasses import replace
from itertools import groupby
from .contracts import *
from .preflight import frozen_inputs, check_settings
from src.rag import materialize_phase1_interface as interface
from src.rag import run_rag_retrieval as retrieval
from src.rag import run_rag_retrieval_v3 as retrieval_v3
from src.rag import build_evidence_index_v2 as index
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import build_trusted_contexts as contexts


def query_record(spec,features,ordinal,values,probabilities):
    row=interface.observation(spec,features,ordinal,values,probabilities,source_split='test')
    row={key:value for key,value in row.items() if not key.startswith('prob_class_')}
    row.update(retrieval.semantic_profile(spec.name,row['predicted_class_label'],row['traffic_representation_json']))
    row.update(classifier_prediction=row['predicted_class_label'],classifier_confidence=row['confidence'],
               classifier_probability_vector=row['class_probabilities_json'],evaluation_scope=SCOPE)
    query_check(row)
    return row


def stage1(root):
    """Copy only frozen safe observations; never reopen traffic rows or labels."""
    before=begin(root,1,frozen_inputs(root))
    if before is None: return
    selected,excluded=cohort(root); expected={r['query_id']:r for r in selected}; found={}
    for row in rows(root,LEGACY_QUERIES):
        q=row['query_id']
        if q not in expected: continue
        require(q not in found,'Duplicate frozen TEST observation')
        require(all(row[k]==expected[q][k] for k in ('source_dataset','classifier_prediction','source_split')),'Frozen prediction differs')
        row=dict(row,evaluation_scope=SCOPE)
        query_check(row); found[q]=row
    require(set(found)==set(expected),'Missing frozen observations')
    table(root,rel(1),(found[r['query_id']] for r in selected))
    manifest=rel(1,'cohort_manifest.csv'); omitted=rel(1,'excluded_query_ids.csv')
    table(root,manifest,selected)
    table(root,omitted,(dict(r,exclusion_reason='Tor prediction ordinal exceeds 50 in frozen query-ID ordering') for r in excluded))
    from .summaries import stage_summaries
    finish(root,1,before,[rel(1),manifest,omitted,*stage_summaries(root,1)],TOTAL)


def references(root,stage):
    require(stage in (2,3),'Reference stage must be 2 or 3')
    before=begin(root,stage,frozen_inputs(root))
    if before is None:
        return
    cfg=config(root,INDEX_CONFIG)
    chunk_rows=list(rows(root,CHUNKS))
    require(len(chunk_rows)==int(cfg['chunk_count']) and all(int(r['chunk_token_count'])<=512 for r in chunk_rows),'Index chunk/token contract differs')
    outputs=[rel(stage)]
    if stage==2:
        docs=list(rows(root,DOCUMENTS))
        table(root,rel(stage),[dict(canonical_corpus=DOCUMENTS,canonical_source_manifest=MANIFEST,
              document_count=len(docs),evaluation_scope=scope(stage))])
        name=rel(stage,'corpus_document_summary.csv')
        table(root,name,({k:r[k] for k in ('document_id','source_type')} for r in docs)); outputs.append(name)
        name=rel(stage,'corpus_curation_summary.csv')
        groups(root,name,QUALITY,('quality_decision',),stage); outputs.append(name)
    else:
        table(root,rel(stage),[dict(canonical_chunks=CHUNKS,index_configuration=INDEX_CONFIG,chunk_count=len(chunk_rows),
              document_count=len({r['document_id'] for r in chunk_rows}),embedding_dimension=cfg['embedding_dimension'],
              tfidf_rows=cfg['chunk_count'],tfidf_columns=cfg['lexical_feature_count'],token_limit=512,evaluation_scope=scope(stage))])
        name=rel(stage,'index_chunk_summary.csv')
        table(root,name,({k:r[k] for k in ('chunk_id','document_id','source_type','chunk_token_count')} for r in chunk_rows)); outputs.append(name)
        for name,fields in (('index_token_distribution.csv',('chunk_token_count',)),('index_document_chunk_counts.csv',('document_id','source_type'))):
            target=rel(stage,name); groups(root,target,CHUNKS,fields,stage); outputs.append(target)
    finish(root,stage,before,outputs,len(chunk_rows) if stage==3 else len(docs))


def candidate_groups(root,stage):
    """Artifacts are emitted in query order; hold at most one candidate group."""
    seen=set()
    for identity,group in groupby(rows(root,rel(stage)),key=lambda r:r['query_id']):
        require(identity not in seen,'Noncontiguous duplicate candidate query')
        seen.add(identity)
        yield identity,list(group)


def stage4(root):
    require_stages(root,1,2,3)
    before=begin(root,4,[rel(1),*frozen_inputs(root)])
    if before is None:
        return
    policy=index.v1.load_token_policy(root)
    evidence=retrieval_v3.load_index(root,policy,DEVELOPMENT_ID)
    encoder=retrieval.load_query_encoder(root,policy)
    require(str(encoder.device).split(':')[0]==config(root,RETRIEVAL_CONFIG)['dense_device'],'Frozen query encoder device differs')
    chunks={r['chunk_id']:r for r in evidence.chunks}
    header=(*retrieval_v3.INTENT_HEADER,'classifier_prediction','source_type','source_split','rag_partition','evaluation_scope')
    count=0
    with writer(root,rel(4),header) as out:
        for batch in retrieval.batches(queries(root),256):
            for query,candidates in retrieval_v3.balanced_batch(evidence,batch,encoder,policy,source_split='test'):
                require(len(candidates)==len(retrieval.METHODS)*retrieval.TOP_K,'Candidate count differs')
                require({(r['retrieval_method'],int(r['rank'])) for r in candidates}=={(m,k) for m in retrieval.METHODS for k in range(1,11)},'Duplicate/missing method ranks')
                require(all(len({r['chunk_id'] for r in candidates if r['retrieval_method']==method})==10 for method in retrieval.METHODS),'Duplicate chunk within retrieval method')
                for row in candidates:
                    require(row['query_id']==query['query_id'] and row['source_dataset']==query['source_dataset'],'Candidate query/dataset differs')
                    chunk=chunks[row['chunk_id']]
                    require(chunk['document_id']==row['document_id'],'Unknown document/chunk linkage')
                    row.update(experiment_id=RUN_ID,classifier_prediction=query['classifier_prediction'],
                               source_type=chunk['source_type'],source_split='test',rag_partition='final_test',evaluation_scope=SCOPE)
                    assert_safe(row); out.writerow(row)
                count+=1
    from .summaries import stage_summaries
    finish(root,4,before,[rel(4),*stage_summaries(root,4)],count)


def stage5(root):
    require_stages(root,1,3,4)
    before=begin(root,5,[rel(4),*frozen_inputs(root)])
    if before is None:
        return
    from .preflight import evidence as frozen_evidence
    _,_,verified=frozen_evidence(root)
    status=rel(5,'provenance_chunk_status.csv'); table(root,status,verified.values())
    count=0
    header=None
    from contextlib import ExitStack
    with ExitStack() as stack:
        out=None
        for identity,candidates in candidate_groups(root,4):
            require('-TEST-' in identity and len(candidates)==30,'Invalid TEST candidate group')
            require(len({(r['retrieval_method'],r['rank']) for r in candidates})==30,'Duplicate candidate rank')
            for r in candidates:
                assert_safe(r)
                v=verified[r['chunk_id']]
                require(r['source_split']=='test' and r['rag_partition']=='final_test' and r['document_id']==v['document_id'],'Provenance TEST linkage mismatch')
                row=dict(r,verification_status=v['provenance_status'],authorization_status=v['authorized_for_downstream'],
                         verification_reason=v['verification_reason'])
                if out is None:
                    out=stack.enter_context(writer(root,rel(5),tuple(row)))
                out.writerow(row)
            count+=1
    from .summaries import stage_summaries
    finish(root,5,before,[rel(5),status,*stage_summaries(root,5)],count)


def alignment(prediction,chunk,policy):
    terms={term.strip().lower().replace('-','_') for key in ('traffic_topics','security_topics') for term in chunk[key].split(';')}
    rules=[r for r in policy if r['predicted_class_label']==prediction]
    require(rules and all(r['uses_ground_truth']=='NO' and r['component_if_matched']=='1' and r['component_if_not_matched']=='0' for r in rules),'Frozen compatibility policy differs')
    return int(any(r['compatible_metadata_term'].lower() in terms for r in rules))


def alignment_parity(root,chunks,policy):
    """Technical extraction of the saved metadata policy must reproduce development values exactly."""
    predicted={r['query_id']:r['predicted_class_label'] for r in rows(root,'results/rag/phase2_dev_retrieval_v3/query_records.csv')}
    count=0
    for r in rows(root,'metadata/rag/task_alignment_v3_hybrid.csv'):
        require(alignment(predicted[r['query_id']],chunks[r['chunk_id']],policy)==int(r['task_alignment_component']),
                'Metadata compatibility adapter differs from frozen development values; no TEST tuning allowed')
        count+=1
    require(count==128724*10,'Incomplete frozen alignment evidence')


def stage6(root):
    require_stages(root,1,4,5)
    before=begin(root,6,[rel(1),rel(5),rel(5,'provenance_chunk_status.csv'),*frozen_inputs(root)])
    if before is None: return
    from .preflight import evidence as frozen_evidence
    _,chunks,verified=frozen_evidence(root)
    from src.rag import build_semantic_association_v5 as method
    claims=method.load_decision_claims(root,DEVELOPMENT_ID)
    saved=config(root,RERANK_CONFIG)
    cfg=dict(method=saved['method'],top_k=int(saved['top_k']),profile='task_relevance',
             source_preferences=json.loads(saved['source_preferences']),
             **{k:float(saved[k]) for k in ('retrieval_weight','source_preference_weight','task_alignment_weight','relationship_weight')})
    query_iter=iter(queries(root)); count=0
    header=(*reranking.TASK_HEADER,'classifier_prediction','rank_delta','task_alignment','reranking_score','verification_status','source_split','rag_partition','evaluation_scope')
    with writer(root,rel(6),header) as out:
        for identity,group in candidate_groups(root,5):
            query=next(query_iter,None); require(query is not None and query['query_id']==identity,'Missing/out-of-order TEST query')
            selected=[r for r in group if r['retrieval_method']==cfg['method']]
            ranked=reranking.rerank(query,selected,verified,cfg,evidence=chunks,decision_claims=claims)
            for row in ranked:
                row.update(classifier_prediction=query['classifier_prediction'],rank_delta=int(row['original_rank'])-int(row['final_rank']),
                           task_alignment=row['task_alignment_component'],reranking_score=row['final_rerank_score'],verification_status=row['provenance_status'],
                           source_split='test',rag_partition='final_test',evaluation_scope=SCOPE)
                assert_safe(row);out.writerow(row)
            count+=1
        require(next(query_iter,None) is None,'Missing reranked queries')
    from .summaries import stage_summaries
    finish(root,6,before,[rel(6),*stage_summaries(root,6)],count)


def stage7(root):
    require_stages(root,1,5,6)
    before=begin(root,7,[rel(1),rel(6),rel(5,'provenance_chunk_status.csv'),*frozen_inputs(root)])
    if before is None:
        return
    chunks={r['chunk_id']:r for r in index.validated_chunks(root,inputs=index.input_paths(DEVELOPMENT_ID),outputs=index.output_paths(DEVELOPMENT_ID))}
    verified={r['chunk_id']:r for r in rows(root,rel(5,'provenance_chunk_status.csv'))}
    policy=index.v1.load_token_policy(root); query_iter=iter(queries(root)); count=0
    detail=rel(7,'trusted_context_chunks.csv')
    with writer(root,rel(7),(*contexts.HEADER,'chunk_count','distinct_document_count','evaluation_scope')) as out, writer(root,detail,('query_id','chunk_id','document_id','source_type','rerank_rank','token_count','provenance_status')) as small:
        for identity,ranked in candidate_groups(root,6):
            query=next(query_iter,None); require(query is not None and query['query_id']==identity,'Missing/out-of-order TEST context query')
            record=contexts.construct(query,ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
            evidence=json.loads(record['evidence_records_json'])
            record.update(chunk_count=len(evidence),distinct_document_count=len({r['document_id'] for r in evidence}),evaluation_scope=SCOPE)
            assert_safe(record); out.writerow(record)
            for item in evidence:
                small.writerow(dict(query_id=identity,**{k:item[k] for k in ('chunk_id','document_id','source_type','rerank_rank','token_count','provenance_status')}))
            count+=1
        require(next(query_iter,None) is None,'Missing TEST contexts')
    from .summaries import stage_summaries
    finish(root,7,before,[rel(7),detail,*stage_summaries(root,7)],count)
