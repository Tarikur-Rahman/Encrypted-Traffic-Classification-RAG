import csv
import io
import json

def export_content(session, format):
    if format == 'json':
        return json.dumps(session, ensure_ascii=False, indent=2, allow_nan=False)
    output = io.StringIO(newline='')
    columns = ['analysis_id','flow_id','dataset','source_dataset','analysis_contract','source_split','predicted_class','confidence','final_class','evidence_status','explanation']
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in session.get('flows', []):
        decision = session.get('phase2', {}).get(row['id'], {}).get('decision', {})
        data = dict(analysis_id=session['analysis_id'], flow_id=row['id'], dataset=session['input']['dataset'],
            source_dataset=row.get('source_dataset',session['input']['dataset']),analysis_contract=session['input']['dataset'],source_split=row.get('source_split',''),
            predicted_class=row['predicted_class'], confidence=row['confidence'], final_class=decision.get('final_class',''),
            evidence_status=decision.get('evidence_status',''), explanation=decision.get('explanation', decision.get('evidence_explanation','')))
        # Spreadsheet export safety; numeric values stay numeric.
        writer.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v) for k,v in data.items()})
    return output.getvalue()
