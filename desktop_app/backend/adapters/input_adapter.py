from pathlib import Path
from backend.config import ROOT, DATASETS, PCAP_UNAVAILABLE
from backend.adapters.artifacts import digest

# The current TEST contract is versioned below its own run directory.  Resolve
# this from the authoritative contract instead of reviving the pre-v5 flat
# ``results/rag/test`` paths.
from src.rag.final_test.contracts import BASE as TEST_BASE

TEST_QUERIES = TEST_BASE + 'stage01_structured_queries/test_query_records.csv'

def sources(dataset):
    prefix = 'ustc' if dataset == DATASETS[0] else 'cic'
    return [
        {'label': 'Completed held-out research results', 'path': str(ROOT / TEST_QUERIES), 'mode': 'saved_test'},
        {'label': 'Saved validation predictions', 'path': str(ROOT / f'results/rag/interface/{prefix}_validation_interface.csv'), 'mode': 'saved_validation'},
        {'label': 'Frozen validation CSV · run existing inference', 'path': str(ROOT / f'data/processed/{dataset}/validation.csv'), 'mode': 'validation_inference'},
    ]

def validate(path, dataset, input_stage='auto'):
    if dataset not in DATASETS:
        raise ValueError('Select a supported dataset contract.')
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError('The selected file cannot be read. Choose an existing local file.')
    extension = source.suffix.lower()
    if extension not in {'.csv', '.pcap', '.pcapng'}:
        raise ValueError('Choose a CSV, PCAP, or PCAPNG file.')
    result = dict(name=source.name, path=str(source), extension=extension, size=source.stat().st_size,
                  dataset=dataset, compatible=False, mode=None)
    if extension != '.csv':
        if dataset == DATASETS[1]:
            return dict(result,status='UNSUPPORTED CONTRACT',reason='CIC-Darknet2020 accepts validated flow-derived CSV input. A validated raw-PCAP → CIC 62-feature extraction contract is not available in the frozen research pipeline.')
        from backend.adapters.contracts import extraction_options
        from src.data.ustc_prepare_dataset import load_streamer
        try:
            with source.open('rb') as stream:
                magic = stream.read(4)
            if magic not in (b'\xd4\xc3\xb2\xa1',b'\xa1\xb2\xc3\xd4',b'\x4d\x3c\xb2\xa1',b'\xa1\xb2\x3c\x4d',b'\x0a\x0d\x0d\x0a'):
                raise ValueError('This file does not contain a recognized PCAP/PCAPNG header.')
            options = extraction_options()
            load_streamer()
        except Exception as error:
            return dict(result,status='UNAVAILABLE',reason=str(error))
        return dict(result,compatible=True,status='READY',mode='external_inference',input_stage='raw_flows',
                    source_dataset='external',analysis_contract=dataset,extractor_configuration=options,
                    reason='USTC native extraction and frozen preprocessing available. Actual flow schema is validated after extraction.')
    for candidate in sources(dataset):
        canonical = Path(candidate['path'])
        if canonical.exists() and (source == canonical.resolve() or
                (source.stat().st_size == canonical.stat().st_size and digest(source) == digest(canonical))):
            return dict(result, compatible=True, status='READY',mode=candidate['mode'], canonical=str(canonical),
                        reason='Exact authoritative artifact match. No preprocessing is fitted or rewritten.')
    from backend.adapters.contracts import contract
    value = contract(dataset)
    features = value.predictor_names
    import csv
    with source.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    from collections import Counter
    duplicates = {name for name,count in Counter(header).items() if count>1}
    # The original CIC reader uses pandas' Label / Label.1 disambiguation.
    # Both are excluded metadata; no predictor aliases are introduced.
    known_cic_labels = dataset==DATASETS[1] and duplicates=={'Label'} and header.count('Label')==2 and 'Label.1' not in header
    if not header or (duplicates and not known_cic_labels):
        return dict(result,status='INCOMPATIBLE',reason='CSV headers must be present and unique.')
    missing = [f for f in features if f not in header]
    matches = dict(required_predictors=len(features),matched_predictors=len(features)-len(missing),missing_features=missing,
                   analysis_contract=dataset,source_dataset='external',columns=header)
    if missing:
        return dict(result,**matches,status='PARTIAL MATCH' if len(missing)<len(features) else 'INCOMPATIBLE',
                    reason='Missing required predictors: '+', '.join(missing))
    raw_match = set(value.raw_names).issubset(header)
    if input_stage == 'auto':
        # Identical column names cannot establish scaling. Require an explicit
        # stage declaration even when all raw candidate columns are present.
        return dict(result,**matches,status='STAGE REQUIRED',input_stage='auto',
                    available_stages=['raw_flows','model_features'] if raw_match else ['model_features'],
                    reason='Schema matches. Declare raw flow measurements or already frozen-standardized model features; values are never guessed.')
    if input_stage not in {'raw_flows','model_features'}:
        raise ValueError('Invalid input stage.')
    if input_stage == 'raw_flows' and not raw_match:
        missing_raw = [f for f in value.raw_names if f not in header]
        return dict(result,**matches,status='PARTIAL MATCH',reason='Missing raw-flow columns: '+', '.join(missing_raw))
    return dict(result,**matches,compatible=True,status='READY',mode='external_inference',input_stage=input_stage,
                reason=f'{len(features)}/{len(features)} predictors matched. {input_stage.replace("_"," ")} declared; all rows will be validated before inference.')

def inspect(path, input_stage='auto'):
    results = [validate(path,dataset,input_stage) for dataset in DATASETS]
    return dict(name=results[0]['name'],path=results[0]['path'],extension=results[0]['extension'],size=results[0]['size'],contracts=results)
