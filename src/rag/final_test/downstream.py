"""Fixed historical V5 TEST fusion/final projection and post-Stage10 evaluation."""
from .contracts import *
from . import llm
from src.evaluation import finalize_v5_contract_v2_stage9 as projection
from src.evaluation import finalize_validation_phase2 as final_projection


def stage9(root):
    require_stages(root,1,7,8)
    before=begin(root,9,[rel(1),rel(7),rel(8),llm.AUDIT,rel(8,CHECKS[7])])
    if before is None:return
    accepted=llm.completed(root)
    require(len(accepted)==TOTAL and all(r['validation_status']=='PASS' for r in accepted.values()),
            'Stage9 blocked: rejected responses remain rejected; no conversion')
    ids=[r['query_id'] for r in cohort(root)[0]]
    contexts=llm.unique_contexts(root,ids); query_map={r['query_id']:r for r in queries(root)}
    result=[]
    for q in ids:
        context=contexts[q]; query=query_map[q]
        require(context['classifier_prediction']==query['classifier_prediction']
                and context['source_dataset']==query['source_dataset'],'Frozen prediction changed')
        row=projection.fuse_accepted(context,accepted[q],source_split='test',
                                    response_reference=llm.EXECUTION+llm.runtime.VALIDATED)
        row.update(decision_source='retain_classifier',supported_class=row['llm_supported_class'],evaluation_scope=SCOPE,run_id=RUN_ID,
                   population_test_queries=sum(SOURCE_COUNTS.values()),evaluated_llm_test_queries=TOTAL,
                   sampling_seed=42,sampling_strategy=STRATEGY)
        assert_safe(row);result.append(row)
    table(root,rel(9),result)
    from .summaries import stage_summaries
    finish(root,9,before,[rel(9),*stage_summaries(root,9)],len(result))


def stage10(root):
    require_stages(root,9)
    before=begin(root,10,[rel(9),rel(9,CHECKS[8])])
    if before is None:return
    result=[]
    for row in rows(root,rel(9)):
        assert_safe(row)
        result.append(final_projection.finalize(row,source_split='test',response_reference=rel(9)))
    ids=[r['query_id'] for r in cohort(root)[0]]
    require([r['query_id'] for r in result]==ids,'Missing/duplicate/reordered final IDs')
    table(root,rel(10),result)
    from .summaries import stage_summaries
    finish(root,10,before,[rel(10),*stage_summaries(root,10)],len(result))


def evaluate(root):
    # This is the sole prediction/GT join. It cannot run before the immutable
    # Stage-10 receipt, and writes only separate evaluation-only artifacts.
    require_stages(root,*range(1,11))
    before=snapshot(root,[rel(10),GT_PATH,rel(10,CHECKS[9])])
    receipt_name=rel(10,'test_evaluation_validation_checklist.csv')
    if (root/receipt_name).exists():
        receipt=list(rows(root,receipt_name,evaluation=True))[0]
        require(receipt['input_metadata_json']==json_text(before),'Post-hoc evaluation inputs changed')
        old=json.loads(receipt['output_metadata_json'])
        require(snapshot(root,[r['path'] for r in old])==old,'Evaluation output changed')
        return
    final={r['query_id']:r for r in rows(root,rel(10))}
    evaluation=[]; seen=set(); populations=Counter()
    for truth in rows(root,GT_PATH,evaluation=True):
        identity=truth['query_id']
        require(identity not in seen and truth['source_split']=='test' and '-TEST-' in identity,'Duplicate/non-TEST ground truth')
        seen.add(identity); populations[truth['source_dataset']]+=1
        if identity not in final:
            continue
        r=final[identity]
        require(r['source_dataset']==truth['source_dataset'],'Evaluation dataset mismatch')
        labels=CLASSES[r['source_dataset']]
        require(labels[int(truth['true_class_id'])]==truth['true_class_label'],'Evaluation class order mismatch')
        evaluation.append(dict(query_id=identity,source_dataset=r['source_dataset'],true_class_label=truth['true_class_label'],
            final_class=r['final_class'],correct=int(r['final_class']==truth['true_class_label']),
            classifier_confidence=r['classifier_confidence'],classifier_probabilities=r['classifier_probabilities'],
            evidence_status=r['evidence_status'],evaluation_scope=SUBSET_SCOPE,
            sampling_caveat='Prediction-stratified subset; unweighted metrics do not estimate natural TEST prevalence'))
    require(dict(populations)==SOURCE_COUNTS and len(evaluation)==TOTAL,'Missing full TEST GT or selected GT match')
    import numpy as np
    from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,balanced_accuracy_score,
        matthews_corrcoef,confusion_matrix,roc_auc_score,average_precision_score)
    metrics=[]; per_class=[]; matrices=[]
    for dataset,labels in CLASSES.items():
        data=[r for r in evaluation if r['source_dataset']==dataset]
        y=np.array([labels.index(r['true_class_label']) for r in data]); pred=np.array([labels.index(r['final_class']) for r in data])
        values=dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),mcc=matthews_corrcoef(y,pred))
        for average in ('macro','weighted'):
            precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=range(len(labels)),average=average,zero_division=0)
            values.update({average+'_precision':precision,average+'_recall':recall,average+'_f1':f1})
        if len(labels)==2:
            pr,rc,f1,_=precision_recall_fscore_support(y,pred,average='binary',pos_label=1,zero_division=0)
            values.update(precision=pr,recall=rc,f1=f1)
        probability=np.array([[float(json.loads(r['classifier_probabilities'])[c]) for c in labels] for r in data])
        require(np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
                and np.allclose(probability.sum(1),1,atol=1e-8,rtol=0),'Invalid canonical TEST vectors')
        if set(y)==set(range(len(labels))):
            if len(labels)==2:
                values['roc_auc']=roc_auc_score(y,probability[:,1]);values['average_precision']=average_precision_score(y,probability[:,1])
            else:
                onehot=np.eye(len(labels))[y]
                values['macro_ovr_roc_auc']=float(np.mean([roc_auc_score(onehot[:,j],probability[:,j]) for j in range(len(labels))]))
                values['macro_average_precision']=float(np.mean([average_precision_score(onehot[:,j],probability[:,j]) for j in range(len(labels))]))
        metrics.extend(dict(source_dataset=dataset,metric=k,value=float(v),sample_size=len(data),evaluation_scope=SUBSET_SCOPE,
                            averaging_note='Unweighted prediction-stratified sample; AP is not trapezoidal PR-AUC') for k,v in values.items())
        pr,rc,f1,support=precision_recall_fscore_support(y,pred,labels=range(len(labels)),zero_division=0)
        per_class.extend(dict(source_dataset=dataset,class_label=c,precision=pr[j],recall=rc[j],f1=f1[j],support=int(support[j]),evaluation_scope=SUBSET_SCOPE) for j,c in enumerate(labels))
        cm=confusion_matrix(y,pred,labels=range(len(labels)))
        matrices.extend(dict(source_dataset=dataset,actual_class=a,predicted_class=b,count=int(cm[j,k]),evaluation_scope=SUBSET_SCOPE)
                        for j,a in enumerate(labels) for k,b in enumerate(labels))
    # Dataset-qualified classes keep the two vocabularies distinct.
    y=[r['source_dataset']+'/'+r['true_class_label'] for r in evaluation]
    predicted=[r['source_dataset']+'/'+r['final_class'] for r in evaluation]
    vocabulary=[d+'/'+label for d,labels in CLASSES.items() for label in labels]
    overall=dict(accuracy=accuracy_score(y,predicted),total_support=len(evaluation))
    for average in ('macro','weighted'):
        precision,recall,f1,_=precision_recall_fscore_support(y,predicted,labels=vocabulary,average=average,zero_division=0)
        overall.update({average+'_precision':precision,average+'_recall':recall,average+'_f1':f1})
    metrics.extend(dict(source_dataset='OVERALL',metric=k,value=float(v),sample_size=len(evaluation),
                        evaluation_scope=SCOPE,averaging_note='Fixed historical prediction-stratified cohort; six dataset-qualified classes; not population weighted')
                   for k,v in overall.items())
    outputs=[]
    for name,data in (('test_evaluation_records',evaluation),('test_evaluation_metrics',metrics),
                      ('test_evaluation_per_class_metrics',per_class),('test_evaluation_confusion_matrix',matrices)):
        target=rel(10,name+'.csv');table(root,target,data);outputs.append(target)
    for name,fields in (('test_evaluation_correctness_by_evidence_status',('source_dataset','evidence_status','correct')),
                        ('test_evaluation_confidence_correctness',('source_dataset','confidence_bin','correct'))):
        counts=Counter()
        for row in evaluation:
            r=dict(row,confidence_bin=min(99,int(float(row['classifier_confidence'])*100)))
            counts[tuple(str(r[k]) for k in fields)]+=1
        target=rel(10,name+'.csv');table(root,target,(dict(zip(fields,key),count=n,evaluation_scope=SUBSET_SCOPE) for key,n in sorted(counts.items())))
        outputs.append(target)
    require(snapshot(root,[r['path'] for r in before])==before,'Scientific outputs changed during post-hoc evaluation')
    table(root,receipt_name,[dict(status='PASS',evaluation_scope=SUBSET_SCOPE,input_metadata_json=json_text(before),output_metadata_json=json_text(snapshot(root,outputs)))])
