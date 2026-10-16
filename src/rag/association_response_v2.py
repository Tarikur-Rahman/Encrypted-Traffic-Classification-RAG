"""Structured linkage and admissible-state decoding; frozen V5 gate unchanged.

The model's narrative is retained verbatim, not certified by the claim validator.
Legacy `explanation` is a linkage encoding, not a rewritten narrative.
"""
from src.rag import run_v5_diagnostic as runtime

common=runtime.common
require=common.require
FIELDS=runtime.validator.FIELDS | {"claim_ids"}
VERSION="v5_association_response_v2"
FROZEN_VALIDATOR_SHA256="9b47110801b0bd9a6a689f4d419ff28d99a257429e8d0724653d722a6a78caed"
PATTERNS=("NO_QUALIFYING","PREDICTED_SUPPORT_ONLY","ALTERNATIVE_ONLY","BOTH_DIRECTIONS")
DISCLOSURE=("Contract v2 was designed after observing systematic contract-v1 validation failures in the fixed historical validation cohort. "
            "It is therefore a development/runtime repair and is not an unbiased post-hoc validation result.")


def admissible(payload,registry,verified,*,source_split="validation"):
    matches=runtime.validator.qualifying(payload,registry,verified,method=runtime.method,source_split=source_split)
    classes=sorted({registry[k]["supported_class"] for k in matches})
    prediction=payload["classifier_prediction"]
    predicted=prediction in classes
    alternative=bool(set(classes)-{prediction})
    pattern="NO_QUALIFYING" if not classes else "BOTH_DIRECTIONS" if predicted and alternative else "PREDICTED_SUPPORT_ONLY" if predicted else "ALTERNATIVE_ONLY"
    if not classes: status,supported,decision="INSUFFICIENT",None,"ABSTAIN"
    elif len(classes)>1: status,supported,decision="CONFLICTING",None,"ABSTAIN"
    else: status,supported,decision=("SUPPORTED" if predicted else "CONTRADICTED"),classes[0],"SUPPORT_CLASS"
    return dict(pattern=pattern,evidence_status=status,supported_class=supported,reasoning_decision=decision,
                classes=classes,claim_ids=sorted(matches),claim_chunks={k:sorted(v) for k,v in sorted(matches.items())})


def schema(state,query_id):
    ids=state["claim_ids"]
    citations=sorted({c for values in state["claim_chunks"].values() for c in values})
    def array(values,minimum):
        item=dict(type="string")
        if values: item["enum"]=values
        return dict(type="array",items=item,minItems=minimum,maxItems=len(values))
    explanation=dict(type="string",minLength=1)
    if not ids: explanation["enum"]=["No qualifying approved claim."]
    return dict(type="object",additionalProperties=False,required=sorted(FIELDS),properties=dict(
        query_id=dict(type="string",enum=[query_id]),
        reasoning_decision=dict(type="string",enum=[state["reasoning_decision"]]),
        evidence_status=dict(type="string",enum=[state["evidence_status"]]),
        supported_class=dict(type="null" if state["supported_class"] is None else "string",enum=[state["supported_class"]]),
        explanation=explanation,claim_ids=array(ids,len(state["classes"]) if ids else 0),
        cited_chunk_ids=array(citations,1 if ids else 0)))


def prompt(payload,registry,verified,state):
    fields=("claim_id","canonical_feature","semantic_state","direction","supported_class","contrasted_class","reference_classes")
    graph=[dict({k:registry[identity][k] for k in fields},cited_chunk_ids=state["claim_chunks"][identity]) for identity in state["claim_ids"]]
    supplied={i for e in payload["trusted_evidence"] for i in common.parse_json(e.get("decision_claim_ids_json","[]"))}
    contextual=sorted(supplied-set(state["claim_ids"]))
    instructions=("Return exactly seven fields: query_id, reasoning_decision, evidence_status, supported_class, explanation, claim_ids, cited_chunk_ids. "
        "Use only supplied evidence. TRAIN associations are exploratory, noncausal and do not establish the true flow class or classifier correctness. "
        "The only admissible evidence status under the verified supplied evidence graph is "+state["evidence_status"]+". "
        "Use the registered decision and supported_class shown in ADMISSIBLE_STATE. Generate a concise visible explanation, not hidden reasoning. "
        "When claims qualify, select one or more allowed claim IDs as a sorted, unique JSON array in claim_ids, not a JSON array embedded in explanation. "
        "For CONFLICTING, select claims covering EVERY distinct supported class in the admissible graph; do not cherry-pick. "
        "Citations must be unique and exactly the union of chunks linked to selected claims. "
        "With INSUFFICIENT, claim_ids and cited_chunk_ids must be empty, supported_class null, decision ABSTAIN, "
        "and explanation exactly 'No qualifying approved claim.'. "
        "Contextual evidence is retained in INPUT_JSON but cannot establish decision support. Unrelated pairs never contradict the evaluated class. "
        "No aliases, confidence, probabilities, true labels or outside knowledge may establish evidence. "
        "Your narrative is preserved verbatim; the validator independently checks structured linkage and does not certify narrative assertions. ")
    return (instructions+"\nADMISSIBLE_STATE:\n"+common.json_text(state)+"\nQUALIFYING_CLAIMS:\n"+common.json_text(graph)
            +"\nCONTEXTUAL_ONLY_CLAIM_IDS:\n"+common.json_text(contextual)+"\nINPUT_JSON:\n"+common.json_text(payload))


def to_legacy(value):
    require(isinstance(value,dict) and set(value)==FIELDS,"Exact seven-field v2 response required")
    require(isinstance(value["claim_ids"],list) and all(isinstance(i,str) for i in value["claim_ids"]),"claim_ids must be a string array")
    require(isinstance(value["explanation"],str) and bool(value["explanation"].strip()),"Model explanation required")
    result={k:value[k] for k in runtime.validator.FIELDS}
    if value["evidence_status"]=="INSUFFICIENT":
        require(not value["claim_ids"],"Abstention cannot carry claims")
        # Preserve the model's exact abstention text; the frozen validator checks it.
    else:
        # JSON serialization preserves list order and duplicates. No repair.
        result["explanation"]="CLAIM_IDS: "+common.json_text(value["claim_ids"])
    return result
