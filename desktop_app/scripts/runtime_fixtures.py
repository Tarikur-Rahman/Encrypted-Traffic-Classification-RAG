"""Small real input slices for desktop smoke tests; no synthetic predictions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import ROOT, RUNTIME
from backend.adapters.artifacts import rows
from backend.adapters.contracts import contract
from backend.utils.paths import output_path
import pandas as pd

def main():
    directory=output_path(RUNTIME/'regression-inputs')
    directory.mkdir(exist_ok=True)
    for dataset in ('USTC-TFC2016','CIC-Darknet2020'):
        value=contract(dataset)
        path=ROOT/('data/interim/USTC-TFC2016/flows.csv' if value.spec.prefix=='ustc' else 'data/raw/CIC-Darknet2020/Darknet.CSV')
        pd.read_csv(path,nrows=3,float_precision='round_trip')[list(value.raw_names)].to_csv(directory/f'{value.spec.prefix}-raw.csv',index=False)
        pd.read_csv(ROOT/f'data/processed/{dataset}/validation.csv',nrows=3,float_precision='round_trip')[list(value.predictor_names)].to_csv(directory/f'{value.spec.prefix}-features.csv',index=False)
    (directory/'incompatible.csv').write_text('unknown\n1\n',encoding='utf-8')

if __name__=='__main__':
    main()
