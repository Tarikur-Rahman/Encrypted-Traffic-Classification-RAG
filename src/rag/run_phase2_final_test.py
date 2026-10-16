"""Final held-out TEST pipeline. Read-only preflight/status unless --execute."""
import argparse
import sqlite3
from pathlib import Path
from src.rag.final_test import contracts as c

STAGE_NAMES=('preflight','queries','references','corpus','index','retrieval','provenance','reranking','contexts',
             'upstream','subset','llm','llm-export','token-audit','llm-audit','fusion','finalize','evaluate','plotting','closure','status','all')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--stage',choices=STAGE_NAMES,default='preflight')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--repair-trailing-partial',action='store_true')
    parser.add_argument('--reset-transport-circuit',action='store_true')
    args=parser.parse_args(argv)
    root=args.project_root.resolve()
    try:
        if args.stage=='status':
            for stage,title in enumerate(c.STAGES,1):
                print(f'{stage} {title}: '+('PASS' if c.committed(root,stage) else 'INCOMPLETE')+f' [{c.scope(stage)}]')
            complete=(root/(c.BASE+'closure/phase2_final_test_closure.csv')).exists()
            if complete:
                c.validate_closure(root)
            print('FINAL_TEST_CLOSURE='+('PASS' if complete else 'INCOMPLETE'))
            return 0
        from src.rag.final_test import preflight
        if args.stage=='llm-audit' or (args.stage=='token-audit' and not args.execute):
            from src.rag.final_test import llm
            llm.audit(root,publish=False)
            return 0
        if args.stage=='preflight' or not args.execute:
            report=preflight.check(root)
            print(c.json_text(report))
            print('PHASE2_FINAL_TEST_PREFLIGHT=PASS\nTEST_COHORT=4300\nNO_NETWORK=YES\nNO_WRITES=YES')
            return 0
        preflight.check_settings(root)
        if args.stage in ('queries','upstream','all'):
            preflight.check(root,scan_queries=False)
        from src.rag.final_test import upstream, llm, downstream, summaries
        actions={'queries':lambda:upstream.stage1(root),'corpus':lambda:upstream.references(root,2),
                 'index':lambda:upstream.references(root,3),'retrieval':lambda:upstream.stage4(root),
                 'provenance':lambda:upstream.stage5(root),'reranking':lambda:upstream.stage6(root),
                 'contexts':lambda:upstream.stage7(root),'subset':lambda:llm.subset(root),
                 'llm':lambda:llm.run(root,resume=args.resume,repair=args.repair_trailing_partial,reset=args.reset_transport_circuit),
                 'token-audit':lambda:llm.audit(root,publish=True),
                 'llm-export':lambda:llm.run(root,export=True),'fusion':lambda:downstream.stage9(root),
                 'finalize':lambda:downstream.stage10(root),'evaluate':lambda:downstream.evaluate(root),
                 'plotting':lambda:summaries.plotting(root),'closure':lambda:c.manifests(root,closure=True)}
        upstream_order=('queries','corpus','index','retrieval','provenance','reranking','contexts')
        sequence=upstream_order if args.stage=='upstream' else ('corpus','index') if args.stage=='references' else (
            (*upstream_order,'subset','token-audit','llm','llm-export','fusion','finalize','evaluate','plotting','closure') if args.stage=='all' else (args.stage,))
        with c.execution_lock(root):
            for name in sequence:
                actions[name]()
                c.manifests(root)
                print('PHASE2_FINAL_TEST_STAGE='+name+' COMPLETED',flush=True)
        return 0
    except KeyboardInterrupt:
        print('PHASE2_FINAL_TEST=INCOMPLETE; durable LLM responses retained for resume')
        return 130
    except (OSError,ValueError,KeyError,ImportError,RuntimeError,sqlite3.Error) as exc:
        print('PHASE2_FINAL_TEST=BLOCKED\n'+str(exc))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
