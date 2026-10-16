"""WSL entry point for the user-supplied local server. No research batch stages."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    root=Path(sys.argv[1]).resolve(strict=True)
    app=Path(__file__).resolve().parents[1]
    if root!=app.parent:
        raise ValueError('The model root must be this desktop checkout.')
    sys.path.insert(0,str(root))
    from src.rag.final_test.contracts import FROZEN
    from src.rag.run_v5_diagnostic import ENDPOINT, MODEL
    if MODEL!='provenance-qwen3-4b' or ENDPOINT!='http://127.0.0.1:8000/v1/chat/completions':
        raise ValueError('Frozen model/server identity differs.')
    args=['vllm','serve',str(root/'models/rag/llm/Qwen3-4B-Instruct-2507-FP8'),
        '--served-model-name',MODEL,'--host','127.0.0.1','--port','8000',
        '--max-model-len',str(FROZEN['max_model_len']),'--gpu-memory-utilization','0.83','--max-num-seqs','1',
        '--enforce-eager','--generation-config','vllm','--seed','42']
    print('Desktop local server:',args,flush=True)
    os.execvp(args[0],args)

if __name__=='__main__':
    main()
