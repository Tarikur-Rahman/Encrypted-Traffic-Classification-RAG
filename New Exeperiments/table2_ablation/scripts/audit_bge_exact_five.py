"""Read-only source audit; write diagnostics only within Table 2. No inference."""
from run_recovered_variants import *

def main():
    ids,queries,original,candidates,verified,chunks,claims,registry,rank_cfg,frozen=load()
    identity='CIC-TEST-030839'
    policy=context.index.v1.load_token_policy(ROOT)
    tokenizer=contract.runtime.qwen.load_tokenizer(ROOT)
    retrieved=sorted((r for r in candidates[identity] if r['retrieval_method']=='dense_bge'),key=lambda r:int(r['rank']))
    ranked=ranking.rerank(queries[identity],retrieved,verified,dict(rank_cfg,method='dense_bge'),evidence=chunks,decision_claims=claims)
    ranked=[{k:str(v) for k,v in r.items()} for r in ranked]
    record=context.construct(queries[identity],ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
    selected=json.loads(record['evidence_records_json'])
    assert len(selected)==4 and record['context_token_count']==1782
    def evidence(row):
        c=chunks[row['chunk_id']]
        return dict(chunk_id=c['chunk_id'],document_id=c['document_id'],source_type=c['source_type'],chunk_text=c['chunk_text'],rerank_rank=int(row['final_rank']),token_count=policy.count(c['chunk_text']),provenance_status='VERIFIED',**{k:row[k] for k in ('decision_claim_match_count','decision_claim_ids_json','decision_claim_classes_json','decision_claim_features_json')})
    ev={r['chunk_id']:evidence(r) for r in ranked}
    def audit(label,items):
        p=payload(record);p['trusted_evidence']=items
        state=contract.admissible(p,registry,verified,source_split='test')
        prompt=contract.prompt(p,registry,verified,state)
        encode=lambda s:len(tokenizer.encode(s,add_special_tokens=False))
        total=contract.runtime.qwen.prompt_tokens(tokenizer,prompt)
        evidence_json=contract.common.json_text(items)
        query={k:v for k,v in p.items() if k!='trusted_evidence'}
        body_counts=[policy.count(e['chunk_text']) for e in items]
        context_tokens=policy.count(context.context_text(items))
        return dict(label=label,chunk_ids=[e['chunk_id'] for e in items],chunk_body_bge_tokens=body_counts,
            context_bge_tokens=context_tokens,context_header_separator_net_overhead=context_tokens-sum(body_counts),trusted_context_budget=2048,
            within_context_budget=context_tokens<=2048,qwen_evidence_json_tokens=encode(evidence_json),qwen_query_fields_json_tokens=encode(contract.common.json_text(query)),
            qwen_full_prompt_tokens=total,qwen_prompt_overhead_excluding_evidence_json=total-encode(evidence_json),
            qwen_chat_wrapper_tokens=total-encode(prompt),explicit_system_message_tokens=0,
            qwen_reserved_completion_tokens=1024,qwen_total_with_completion_reservation=total+1024,qwen_model_limit=5120)
    cases=[audit('BGE retrieval ranks 1-5 (diagnostic only)',[ev[r['chunk_id']] for r in retrieved[:5]]),
        audit('BGE downstream rerank ranks 1-5 (diagnostic only)',[ev[r['chunk_id']] for r in ranked[:5]]),
        audit('Unchanged registered builder output (four; not eligible for inference)',selected),
        audit('Frozen Full system (five)',json.loads(original[identity]['evidence_records_json']))]
    remaining=[ev[r['chunk_id']] for r in ranked if r['chunk_id'] not in {e['chunk_id'] for e in selected}]
    cases += [audit('Selected four plus '+e['chunk_id']+' (diagnostic only)',sorted(selected+[e],key=lambda x:x['rerank_rank'])) for e in remaining]
    assert all(not c['within_context_budget'] for c in cases[4:])
    result=dict(query_id=identity,status='BLOCKED_EXACT_FIVE_CONTEXT_BUDGET',registered_context_budget=2048,required_final_chunks=5,
        candidates=[dict(retrieval_rank=int(r['rank']),rerank_rank=ev[r['chunk_id']]['rerank_rank'],chunk_id=r['chunk_id'],chunk_body_bge_tokens=policy.count(chunks[r['chunk_id']]['chunk_text'])) for r in retrieved],cases=cases,
        mechanism='Existing evidence_diverse selection skips whole chunks over budget. No registered truncation or repacking mechanism preserves exactly five. Full used this same unchanged construct; all 188 Full contexts and prompt hashes reproduced exactly.',
        accounting='2048 bounds BGE-tokenized context_text including evidence headers, not Qwen prompt. BGE bodies each include special tokens; net overhead subtracts those repeated special tokens. Qwen overhead is an exact difference; separately encoded components are not additive. One user message, no explicit system message. Diagnostic five-chunk prompts were never sent to the LLM.',
        source_evidence=['src/rag/build_trusted_contexts.py:construct and context_text','src/rag/build_evidence_index.py:TokenPolicy.count and token_windows','src/rag/build_evidence_index_v2.py:chunk_units','src/rag/final_test/llm.py','src/rag/run_v5_contract_v2.py','src/rag/association_response_v2.py:prompt','data/recovered_configs/stage07_trusted_contexts/trusted_context_configuration.csv'])
    dump(OUT/'audit/bge_exact_five_token_audit.json',result)
    report='# BGE-only exact-five token-budget audit\n\nQuery: '+identity+'\n\nDecision: BLOCKED. Exactly five final chunks and the 2,048-token trusted-context budget are required. No inference performed for this variant.\n\n'+result['mechanism']+'\n\n'+result['accounting']+'\n\nOffline corpus token_windows/chunk_units create the frozen source chunks before indexing; they are not a runtime truncation mechanism. Other historical response-truncation handling concerns completions, not this registered context pipeline. No new packing, truncation, budget change, or source edit was applied.\n\n## All BGE candidates\n\n| Retrieval rank | Rerank | Chunk ID | Body BGE tokens |\n|---|---|---|---|\n'
    report+=''.join(f"| {r['retrieval_rank']} | {r['rerank_rank']} | {r['chunk_id']} | {r['chunk_body_bge_tokens']} |\n" for r in result['candidates'])
    for c in cases:
        report+='\n## '+c['label']+'\n\n```json\n'+json.dumps(c,indent=2)+'\n```\n'
    report+='\nSource audit: '+', '.join(result['source_evidence'])+'.\n'
    (OUT/'audit/bge_context_budget_report.md').write_text(report,encoding='utf-8')
    directory=OUT/'variant_outputs/bge_retrieval_only'
    preflight=json.loads((directory/'execution_preflight.json').read_text(encoding='utf-8'))
    preflight.update(status=result['status'],blocked_reason='CIC-TEST-030839: registered builder selects four chunks (1782 BGE tokens); every remaining fifth candidate exceeds 2048; no registered exact-five truncation/packing mechanism.',token_budget_audit='audit/bge_exact_five_token_audit.json')
    dump(directory/'execution_preflight.json',preflight)
    manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(run_status='BLOCKED',blocked_reason=preflight['blocked_reason'],fresh_inference=False,fresh_inference_count=0)
    dump(directory/'run_manifest.json',manifest)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
