"""Separate claim-grounded V5 contract. No change to the frozen V4 gate.

Positive statuses describe compatibility with registered TRAIN associations,
not ground truth or certainty. Multiple supported classes are a conflict.
The registry and verified chunks are supplied by the offline evidence audit,
never by the model. Classifier confidence is not an input to this gate.
"""
from src.rag import phase2_common as common
from src.rag import build_train_evidence_v5 as empirical
from src.rag.run_provenance_aware_reranking import PROFILE_DIRECTIONS

FIELDS = frozenset(("query_id", "reasoning_decision", "evidence_status", "supported_class", "explanation", "cited_chunk_ids"))
POLICY = (
    "Return exactly query_id, reasoning_decision, evidence_status, supported_class, explanation, cited_chunk_ids. "
    "Use only supplied approved claim IDs and verbatim evidence. TRAIN associations are noncausal and do not prove a flow label. "
    "SUPPORTED requires one unambiguous claim-supported class equal to classifier_prediction and decision SUPPORT_CLASS. "
    "CONTRADICTED requires one unambiguous alternative class whose registered reference explicitly includes classifier_prediction, "
    "and decision SUPPORT_CLASS. CONFLICTING requires compatible approved claims for at least two different classes, decision ABSTAIN, "
    "supported_class null. INSUFFICIENT uses ABSTAIN, supported_class null, no citations, explanation 'No qualifying approved claim.' "
    "For every other status, explanation must be exactly 'CLAIM_IDS: ' followed by a sorted unique JSON array of claim IDs. "
    "Cite exactly the supplied chunks containing those claims. No aliases, confidence, probabilities, outside knowledge, "
    "rewritten quotes, or contextual fallback may establish support. Contradiction means competing TRAIN-association evidence, "
    "not demonstrated classifier error. Fusion remains retain_classifier."
)


def observations(profile, dataset):
    start = ". Observed traffic profile relative to the training distribution: "
    end = ". Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics."
    common.require(isinstance(profile,str) and profile.count(start)==1 and profile.endswith(end), "Invalid canonical traffic profile")
    result = {}
    for item in profile.split(start,1)[1][:-len(end)].split("; "):
        feature, separator, bin_name = item.rpartition(" is ")
        common.require(separator and feature in empirical.corpus.CANONICAL_DECISION_FEATURES[dataset]
                       and bin_name in PROFILE_DIRECTIONS and feature not in result, "Invalid/duplicate canonical observation")
        result[feature] = PROFILE_DIRECTIONS[bin_name]
    return result


def qualifying(payload, registry, verified, *, method=empirical, source_split="validation"):
    common.require(set(payload)=={"query_id","source_dataset","source_split","classifier_prediction","traffic_profile","trusted_evidence"},
                   "Unexpected V5 runtime input; labels/confidence prohibited")
    dataset = payload["source_dataset"]
    common.require(source_split in {"validation", "test"}, "Invalid explicit V5 split")
    marker, excluded = ("-VAL-", "-TEST-") if source_split == "validation" else ("-TEST-", "-VAL-")
    common.require(dataset in common.CLASSES and payload["source_split"]==source_split
                   and marker in payload["query_id"] and excluded not in payload["query_id"], "V5 split/ID mismatch")
    common.require(payload["classifier_prediction"] in common.CLASSES[dataset], "Invalid hypothesis")
    observed = observations(payload["traffic_profile"], dataset)
    matches, seen = {}, set()
    for item in payload["trusted_evidence"]:
        identity = item["chunk_id"]
        common.require(identity not in seen, "Duplicate supplied chunk")
        seen.add(identity)
        source = verified.get(identity)
        common.require(source is not None and source["provenance_status"]=="VERIFIED"
                       and source["authorized_for_downstream"]=="YES" and source["quality_status"]=="KEEP"
                       and source["document_id"]==item["document_id"] and source["chunk_text"]==item["chunk_text"],
                       "Unverified/altered supplied evidence")
        for claim_id in common.parse_json(item.get("decision_claim_ids_json", "[]")):
            row = registry.get(claim_id)
            common.require(row is not None and row["decision"]=="APPROVED" and row["source_type"]==empirical.SOURCE_TYPE,
                           "Unknown/unapproved claim")
            common.require(row["source_dataset"]==dataset and item["document_id"]==claim_id,
                           "Claim dataset/document mismatch")
            common.require(method.render(row) in item["chunk_text"].splitlines(), "Claim must be a complete verbatim supplied line")
            common.require(row["canonical_feature"] in empirical.corpus.CANONICAL_DECISION_FEATURES[dataset]
                           and observed.get(row["canonical_feature"])==row["direction"], "Observed claim direction differs")
            common.require(row["supported_class"] in common.CLASSES[dataset], "Claim class differs")
            if method.VERSION == "train-semantic-association-v2":
                common.require(row["evidence_method_version"] == method.VERSION
                               and row["semantic_state"] in method.DIRECTIONS
                               and row["direction"] == method.DIRECTIONS[row["semantic_state"]]
                               and row["contrasted_class"] in common.CLASSES[dataset]
                               and row["contrasted_class"] != row["supported_class"]
                               and row["reference_classes"] == [row["contrasted_class"]], "Invalid pairwise association")
                # A VPN-vs-NonVPN contrast cannot contradict Tor. Such evidence
                # stays contextual for an unrelated evaluated hypothesis.
                if payload["classifier_prediction"] not in {row["supported_class"],row["contrasted_class"]}:
                    continue
            matches.setdefault(claim_id, set()).add(identity)
    return matches


def validate(value, payload, registry, verified, *, method=empirical, source_split="validation"):
    common.require(isinstance(value,dict) and set(value)==FIELDS, "Exact six-field V5 response required")
    common.require(value["query_id"]==payload["query_id"], "Wrong query_id")
    common.require(all(isinstance(value[k],str) for k in ("query_id","reasoning_decision","evidence_status","explanation"))
                   and (value["supported_class"] is None or isinstance(value["supported_class"],str))
                   and isinstance(value["cited_chunk_ids"],list)
                   and all(isinstance(c,str) for c in value["cited_chunk_ids"]), "Invalid response types")
    citations = value["cited_chunk_ids"]
    common.require(len(set(citations))==len(citations), "Duplicate citation")
    matches = qualifying(payload,registry,verified,method=method,source_split=source_split)
    status = value["evidence_status"]
    common.require(status in common.STATUSES, "Invalid evidence status")
    if status=="INSUFFICIENT":
        common.require(value["reasoning_decision"]=="ABSTAIN" and value["supported_class"] is None and not citations
                       and value["explanation"]=="No qualifying approved claim.", "Invalid abstention")
        common.require(not matches, "Qualifying claims require an explicit supported/contradicted/conflicting disposition")
        return
    common.require(value["explanation"].startswith("CLAIM_IDS: "), "Exact claim linkage required")
    ids = common.parse_json(value["explanation"][len("CLAIM_IDS: "):])
    common.require(isinstance(ids,list) and ids and all(isinstance(i,str) for i in ids)
                   and ids==sorted(set(ids)) and set(ids)<=set(matches), "Unknown/duplicate/unmatched claim linkage")
    common.require(set(citations)==set().union(*(matches[i] for i in ids)), "Citations do not exactly support linked claims")
    classes = {registry[i]["supported_class"] for i in matches}
    linked_classes = {registry[i]["supported_class"] for i in ids}
    if status=="CONFLICTING":
        common.require(len(classes)>=2 and linked_classes==classes and value["supported_class"] is None
                       and value["reasoning_decision"]=="ABSTAIN", "No genuine approved-claim conflict")
        return
    supported = value["supported_class"]
    common.require(classes==linked_classes=={supported}, "Ambiguous/unsupported positive class")
    prediction = payload["classifier_prediction"]
    if status=="SUPPORTED":
        common.require(supported==prediction and value["reasoning_decision"]=="SUPPORT_CLASS", "Invalid supported relationship")
    else:
        common.require(supported!=prediction and value["reasoning_decision"]=="SUPPORT_CLASS"
                       and all(prediction in registry[i]["reference_classes"] for i in ids), "No explicit competing-class contrast")


def prompt(payload, registry, verified, *, method=empirical):
    matches = qualifying(payload,registry,verified,method=method)
    fields = ("claim_id","canonical_feature","direction","supported_class","reference_classes")
    if method.VERSION == "train-semantic-association-v2":
        fields += ("evidence_method_version","semantic_state","contrasted_class")
    claims = [{k:registry[i][k] for k in fields}
              for i in sorted(matches)]
    policy = POLICY
    if method.VERSION == "train-semantic-association-v2":
        policy += (" Only named pairwise contrasts involving classifier_prediction qualify. LOWER means the four low bins; "
                   "HIGHER means the four high bins; TYPICAL means typical. Unrelated pairwise evidence remains contextual. "
                   "These are exploratory TRAIN associations with marginal, not simultaneous, confidence intervals.")
    return policy+"\nAPPROVED_CLAIMS_JSON:\n"+common.json_text(claims)+"\nINPUT_JSON:\n"+common.json_text(payload)
