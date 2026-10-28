"""Experiment-process-only removal of provenance authorization predicates.

Compile existing functions in memory; never write or copy source implementations.
Keep ID, text, claim linkage, quality, schema, score and token validation intact.
"""
import ast
import inspect
from contextlib import contextmanager
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import build_trusted_contexts as contexts
from src.rag import positive_evidence_v5 as validator
from src.evaluation import finalize_v5_contract_v2_stage9 as fusion

FIELDS={'provenance_status','authorized_for_retrieval','authorized_for_downstream','authorized_for_decision_evidence'}

class AuthorizationOnly(ast.NodeTransformer):
    def __init__(self):self.removed=[]
    def visit_Compare(self,node):
        keys={n.slice.value for n in ast.walk(node) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)}
        if keys & FIELDS:
            self.removed.append(ast.unparse(node))
            return ast.copy_location(ast.Constant(True),node)
        return self.generic_visit(node)
    def visit_Call(self,node):
        if ast.unparse(node.func)=='common.corpus.trusted':
            self.removed.append(ast.unparse(node))
            return ast.copy_location(ast.Constant(True),node)
        return self.generic_visit(node)

@contextmanager
def bypass_provenance():
    targets=[(reranking,'matching_decision_claims'),(reranking,'rerank'),(contexts,'construct'),
             (validator,'qualifying'),(fusion,'fuse_accepted')]
    originals=[];audit=[]
    try:
        for module,name in targets:
            original=getattr(module,name);originals.append((module,name,original))
            transform=AuthorizationOnly()
            tree=transform.visit(ast.parse(inspect.getsource(original)))
            assert transform.removed,(module.__name__,name)
            ast.fix_missing_locations(tree)
            namespace={}
            exec(compile(tree,inspect.getsourcefile(original)+' [Table2 provenance-only adapter]','exec'),module.__dict__,namespace)
            setattr(module,name,namespace[name])
            audit.append(dict(module=module.__name__,function=name,removed_predicates=transform.removed))
        yield audit
    finally:
        for module,name,original in originals:setattr(module,name,original)
