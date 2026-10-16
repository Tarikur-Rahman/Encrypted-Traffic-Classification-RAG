"""One streaming pass per stage creates small, explicit plotting summaries."""
from collections import Counter
from .contracts import *

REPORTS = {
1:{'test_query_summary':('source_dataset',),'test_query_class_distribution':('source_dataset','classifier_prediction'),
   'test_query_provenance_summary':('source_dataset','classifier_run_id','source_split'),
   'test_query_confidence_histogram':('source_dataset','confidence_bin')},
4:{'retrieval_query_summary':('query_id','source_dataset','retrieval_method'),
   'retrieval_method_summary':('retrieval_method',),'retrieval_dataset_summary':('source_dataset','retrieval_method'),
   'retrieval_rank_distribution':('retrieval_method','rank'),'retrieval_document_distribution':('retrieval_method','document_id','rank'),
   'retrieval_source_distribution':('retrieval_method','source_type')},
5:{'provenance_query_summary':('query_id','source_dataset','verification_status','authorization_status'),
   'provenance_document_summary':('document_id','verification_status','authorization_status'),
   'provenance_source_type_summary':('source_type','verification_status','authorization_status'),
   'provenance_status_summary':('source_dataset','verification_status','authorization_status')},
6:{'reranking_query_summary':('query_id','source_dataset'),'reranking_rank_movement':('source_dataset','rank_delta'),
   'reranking_alignment_summary':('classifier_prediction','task_alignment'),
   'reranking_document_distribution':('document_id','final_rank'),'reranking_dataset_summary':('source_dataset',)},
7:{'trusted_context_summary':('context_status',),'trusted_context_token_distribution':('context_token_count',),
   'trusted_context_chunk_distribution':('chunk_count',),'trusted_context_document_diversity':('distinct_document_count',),
   'trusted_context_dataset_summary':('source_dataset','context_status')},
8:{'test_llm_execution_summary':('source_dataset','response_valid'),'test_llm_fallback_summary':('fallback_category',),
   'test_llm_evidence_status_summary':('evidence_status',),'test_llm_reasoning_decision_summary':('reasoning_decision',),
   'test_llm_by_dataset':('source_dataset','evidence_status'),'test_llm_by_stratum':('source_dataset','classifier_prediction','evidence_status'),
   'test_llm_prompt_token_histogram':('prompt_token_bin',),'test_llm_completion_token_histogram':('completion_token_bin',),
   'test_llm_latency_histogram':('latency_bin_seconds',)},
9:{'test_fusion_summary':('decision_source','fusion_reason'),'test_fusion_by_dataset':('source_dataset','decision_source'),
   'test_fusion_by_stratum':('source_dataset','classifier_prediction','evidence_status'),
   'test_fusion_by_evidence_status':('evidence_status','decision_source'),
   'test_fusion_final_class_matrix':('source_dataset','evidence_status','final_class')},
10:{'test_final_classification_summary':('source_dataset','decision_source'),
    'test_final_class_distribution':('source_dataset','final_class'),'test_final_evidence_status_distribution':('evidence_status',),
    'test_final_class_evidence_matrix':('source_dataset','final_class','evidence_status'),
    'test_final_confidence_by_evidence_status':('source_dataset','evidence_status','confidence_bin'),
    'test_final_reasoning_decision_distribution':('reasoning_decision',),'test_final_supported_class_distribution':('supported_class',),
    'test_final_provenance_summary':('provenance_available','explanation_available')},
}


def measured(row):
    value=dict(row)
    if 'classifier_confidence' in row:
        value['confidence_bin']=min(99,int(float(row['classifier_confidence'])*100))
    value['fallback_category']=row.get('fallback_category') or row.get('runtime_fallback_reason') or row.get('validator_fallback_reason') or 'MODEL_NATIVE'
    if 'prompt_tokens' in row:
        transport=row.get('runtime_fallback_reason')=='TRANSPORT_RETRIES_EXHAUSTED'
        value['prompt_token_bin']=int(row['prompt_tokens'])//128*128
        value['completion_token_bin']='UNAVAILABLE' if transport else int(row['completion_tokens'])//64*64
        value['latency_bin_seconds']=int(float(row['latency_seconds'])//5)*5
    if 'provenance' in row:
        value['provenance_available']='YES' if json.loads(row['provenance']) else 'NO'
        value['explanation_available']='YES' if row.get('explanation') else 'NO'
    return value


def stage_summaries(root,stage):
    counters={name:Counter() for name in REPORTS[stage]}
    probability={}
    token_rows=[]; latency=[]
    query_metadata={r['query_id']:r for r in queries(root)} if stage==8 else {}
    for raw in rows(root,rel(stage)):
        if stage==8:
            meta=query_metadata[raw['query_id']]
            value=raw['validated_response']
            raw=dict(raw,source_dataset=meta['source_dataset'],classifier_prediction=meta['classifier_prediction'],
                     response_valid=raw['validation_status']=='PASS',
                     evidence_status=value['evidence_status'] if value else 'REJECTED',
                     reasoning_decision=value['reasoning_decision'] if value else 'REJECTED',
                     fallback_category='MODEL_NATIVE' if value else 'REJECTED',
                     **raw['usage'],retry_count=0)
        row=measured(raw)
        for name,fields in REPORTS[stage].items():
            require(all(k in row for k in fields),'Missing summary field: '+name)
            counters[name][tuple(str(row[k]) for k in fields)]+=1
        if stage==1:
            for label,p in json.loads(row['class_probabilities_json']).items():
                key=(row['source_dataset'],label); v=float(p)
                stat=probability.setdefault(key,[0,0.,1.,0.])
                stat[0]+=1;stat[1]+=v;stat[2]=min(stat[2],v);stat[3]=max(stat[3],v)
        if stage==8:
            token_rows.append({k:row[k] for k in ('query_id','prompt_tokens','completion_tokens','total_tokens','fallback_category')})
            latency.append({k:row[k] for k in ('query_id','latency_seconds','retry_count','fallback_category')})
    outputs=[]
    for name,fields in REPORTS[stage].items():
        target=rel(stage,name+'.csv')
        table(root,target,(dict(zip(fields,key),count=n,evaluation_scope=scope(stage)) for key,n in sorted(counters[name].items())),(*fields,'count','evaluation_scope'))
        outputs.append(target)
    if stage==1:
        target=rel(1,'test_query_probability_summary.csv')
        table(root,target,(dict(source_dataset=d,class_label=c,count=v[0],mean_probability=v[1]/v[0],
                               min_probability=v[2],max_probability=v[3],evaluation_scope=SCOPE) for (d,c),v in sorted(probability.items())))
        outputs.append(target)
    if stage==8:
        for name,records in (('test_llm_token_usage.csv',token_rows),('test_llm_latency.csv',latency)):
            target=rel(8,name); table(root,target,records); outputs.append(target)
    return outputs


PLOTTING = {
'stage01_query_class_counts':(1,'test_query_class_distribution'), 'stage01_confidence_histogram':(1,'test_query_confidence_histogram'),
'stage02_corpus_curation_counts':(2,'corpus_curation_summary'),'stage02_document_counts':(2,'corpus_document_summary'),
'stage03_chunk_token_histogram':(3,'index_token_distribution'),'stage03_chunks_per_document':(3,'index_document_chunk_counts'),
'stage04_retrieval_method_counts':(4,'retrieval_method_summary'),'stage04_document_rank_counts':(4,'retrieval_document_distribution'),
'stage04_source_distribution':(4,'retrieval_source_distribution'),'stage05_provenance_status_counts':(5,'provenance_status_summary'),
'stage05_provenance_source_counts':(5,'provenance_source_type_summary'),'stage06_rank_delta_histogram':(6,'reranking_rank_movement'),
'stage06_task_alignment_counts':(6,'reranking_alignment_summary'),'stage07_context_token_histogram':(7,'trusted_context_token_distribution'),
'stage07_chunk_count_distribution':(7,'trusted_context_chunk_distribution'),'stage07_document_diversity_distribution':(7,'trusted_context_document_diversity'),
'stage08_evidence_status_counts':(8,'test_llm_evidence_status_summary'),'stage08_reasoning_decision_counts':(8,'test_llm_reasoning_decision_summary'),
'stage08_fallback_counts':(8,'test_llm_fallback_summary'),'stage08_prompt_token_histogram':(8,'test_llm_prompt_token_histogram'),
'stage08_completion_token_histogram':(8,'test_llm_completion_token_histogram'),'stage08_latency_histogram':(8,'test_llm_latency_histogram'),
'stage08_by_stratum':(8,'test_llm_by_stratum'),'stage09_decision_source_counts':(9,'test_fusion_summary'),
'stage09_evidence_final_class_matrix':(9,'test_fusion_final_class_matrix'),'stage10_final_class_counts':(10,'test_final_class_distribution'),
'stage10_evidence_status_counts':(10,'test_final_evidence_status_distribution'),'stage10_final_class_evidence_matrix':(10,'test_final_class_evidence_matrix'),
'stage10_correctness_by_evidence_status':(10,'test_evaluation_correctness_by_evidence_status'),
'stage10_per_class_metrics':(10,'test_evaluation_per_class_metrics'),
'stage10_dataset_overall_metrics':(10,'test_evaluation_metrics'),
'stage10_confusion_matrix_values':(10,'test_evaluation_confusion_matrix'),
}


def plotting(root):
    require_stages(root,*range(1,11))
    before=snapshot(root,[rel(stage,name+'.csv') for stage,name in PLOTTING.values()])
    outputs=[]
    for name,(stage,source) in PLOTTING.items():
        target=BASE+'plotting/'+name+'.csv'
        if name=='stage02_document_counts':
            groups(root,target,rel(stage,source+'.csv'),('source_type',),stage)
        else:
            table(root,target,rows(root,rel(stage,source+'.csv'),evaluation=True))
        outputs.append(target)
    require(snapshot(root,[r['path'] for r in before])==before,'Summary sources changed')
    table(root,BASE+'plotting/plotting_validation_checklist.csv',[dict(status='PASS',evaluation_scope=SUBSET_SCOPE,
          input_metadata_json=json_text(before),output_metadata_json=json_text(snapshot(root,outputs)))])
