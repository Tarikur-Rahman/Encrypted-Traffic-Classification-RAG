from backend.config import ROOT
from src.rag import phase2_common as common

IDENTITY=('query_id','source_dataset','analysis_contract','source_split','evaluation_scope')
FORBIDDEN={'true_class_id','true_class_label','classifier_correct','ground_truth','target_id','target_binary'}

def identity(value):
    return {key:value[key] for key in IDENTITY}

def validate_identity(value):
    common.require(value.get('source_dataset')=='external' and value.get('source_split')=='external_inference'
        and value.get('evaluation_scope')=='external_runtime','External runtime identity required')
    common.require(value.get('analysis_contract') in common.CLASSES,'Unknown analysis contract')
    common.require(isinstance(value.get('query_id'),str) and value['query_id'].startswith('EXT-')
        and '-VAL-' not in value['query_id'] and '-TEST-' not in value['query_id'],'External query ID required')
    common.require(not FORBIDDEN.intersection(value),'Ground truth prohibited in external runtime')
    return common.CLASSES[value['analysis_contract']]

def validate_query(query):
    # Independent external scope gate. Never passes external values to the frozen
    # evaluation validator and never patches its validation/test restrictions.
    labels=validate_identity(query)
    common.require(query['predicted_class_label'] in labels,'Invalid predicted class')
    confidence=common.finite(query['confidence'])
    common.require(0<=confidence<=1,'Invalid confidence')
    probabilities=common.parse_json(query['class_probabilities_json'])
    common.require(isinstance(probabilities,dict) and set(probabilities)==set(labels),'Probability classes differ')
    values=[common.finite(probabilities[label]) for label in labels]
    common.require(all(0<=v<=1 for v in values) and abs(sum(values)-1)<=1e-8,'Invalid probability distribution')
    expected=labels[int(values[1]>=.5)] if len(labels)==2 else labels[max(range(len(values)),key=values.__getitem__)]
    common.require(expected==query['predicted_class_label'] and abs(max(values)-confidence)<=1e-8,'Classifier observation differs')
    common.require(isinstance(query['retrieval_query_text'],str) and query['retrieval_query_text'].strip(),'Missing traffic profile')

def same_identity(*values):
    for value in values:
        validate_identity(value)
    common.require(all(identity(value)==identity(values[0]) for value in values),'Query or analysis-contract identity mismatch')
