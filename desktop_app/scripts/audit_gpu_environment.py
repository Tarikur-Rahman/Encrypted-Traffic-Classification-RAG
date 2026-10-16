"""Read-only inspection; run with the existing WSL research interpreter."""
import sys,json,importlib
sys.dont_write_bytecode=True
result={'python_path':sys.executable,'python_version':sys.version,'environment':sys.prefix,'packages':{}}
for name in ['cuml','cupy','xgboost','numpy','pandas']:
    try:
        module=importlib.import_module(name)
        result['packages'][name]=module.__version__
        if name=='cupy':
            result['cuda_runtime']=module.cuda.runtime.runtimeGetVersion()
            result['cuda_driver']=module.cuda.runtime.driverGetVersion()
            result['device_count']=module.cuda.runtime.getDeviceCount()
    except Exception as exc:
        result['packages'][name]={'error':str(exc)}
print(json.dumps(result,indent=2))
