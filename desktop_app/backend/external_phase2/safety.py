import json
from backend.utils.paths import output_path, analysis_directory

def assert_desktop_write_path(path):
    return output_path(path)

def write_stage(analysis_id, flow_number, stage, name, value, *, attempt):
    if stage not in {'trusted_context','llm','fusion'} or name not in {'context','prompt','response','runtime','decision'}:
        raise ValueError('Unknown external runtime artifact.')
    if type(flow_number) is not int or flow_number<0:
        raise ValueError('Invalid flow number.')
    if not isinstance(attempt,str) or not attempt.isalnum():
        raise ValueError('Invalid runtime attempt identity.')
    directory=assert_desktop_write_path(analysis_directory(analysis_id)/'phase2'/stage/str(flow_number)/attempt)
    directory.mkdir(parents=True,exist_ok=True)
    target=assert_desktop_write_path(directory/(name+'.json'))
    temporary=assert_desktop_write_path(target.with_suffix('.tmp'))
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(target)
    return str(target)
