"""Read-only registry and rehydration of the persisted TRAIN transform.

Formula parity is explicitly established by ustc_integrity_audit section F and
cic_preprocess.verify_materialization. No estimator is fitted in this adapter.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
from backend.config import ROOT
from backend.adapters.artifacts import rows, digest

@dataclass(frozen=True)
class DatasetRuntimeContract:
    dataset_id: str
    predictor_names: tuple[str, ...]
    raw_names: tuple[str, ...]
    model_path: Path
    preprocessing_path: Path
    class_names: tuple[str, ...]
    spec: object
    pcap_supported: bool

    @property
    def predictor_count(self):
        return len(self.predictor_names)

@lru_cache(maxsize=2)
def contract(dataset):
    from src.rag import materialize_phase1_interface as interface
    spec = next((s for s in interface.DATASETS if s.name == dataset), None)
    if spec is None:
        raise ValueError('Unknown analysis contract.')
    names = tuple(interface.predictor_order(ROOT, spec))
    path = ROOT / spec.parameters
    parameters = list(rows(path))
    return DatasetRuntimeContract(dataset,names,tuple(r['feature_name'] for r in parameters),
        interface.model_path(ROOT,spec),path,spec.labels,spec,spec.prefix=='ustc')

@lru_cache(maxsize=2)
def preprocessing(dataset):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    value = contract(dataset)
    records = {r['feature_name']:r for r in rows(value.preprocessing_path)}
    mean,scale = ('train_mean','train_std') if value.spec.prefix=='ustc' else ('train_scaler_mean','train_scaler_scale')
    medians = np.array([float(records[f]['train_median']) for f in value.predictor_names])
    scaler = StandardScaler()
    scaler.mean_ = np.array([float(records[f][mean]) for f in value.predictor_names])
    scaler.scale_ = np.array([float(records[f][scale]) for f in value.predictor_names])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = value.predictor_count
    if not np.isfinite(medians).all() or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or not (scaler.scale_>0).all():
        raise ValueError('Invalid frozen preprocessing parameters.')
    return medians,scaler

def transform(frame, value, input_stage):
    import numpy as np
    import pandas as pd
    missing = [f for f in value.predictor_names if f not in frame]
    if missing:
        raise ValueError('Missing required predictors: '+', '.join(missing))
    if input_stage == 'model_features':
        matrix = frame[list(value.predictor_names)].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError('Model-feature CSV must contain finite, already-standardized predictors. Missing values cannot be imputed at this stage.')
        return matrix
    if input_stage != 'raw_flows':
        raise ValueError('Declare raw flows or frozen model features explicitly.')
    missing_raw = [f for f in value.raw_names if f not in frame]
    if missing_raw:
        raise ValueError('Missing raw-flow contract columns: '+', '.join(missing_raw))
    if value.spec.prefix == 'cic':
        from src.data.cic_preprocess import numeric_candidates
        clean = numeric_candidates(frame,list(value.raw_names))[list(value.predictor_names)].to_numpy(dtype=np.float64)
    else:
        clean = frame[list(value.predictor_names)].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
        clean[~np.isfinite(clean)] = np.nan
    medians, scaler = preprocessing(value.dataset_id)
    matrix = scaler.transform(np.where(np.isfinite(clean),clean,medians))
    if not np.isfinite(matrix).all():
        raise ValueError('Frozen transformation produced nonfinite predictors.')
    return matrix

def extraction_options():
    from src.data.ustc_prepare_dataset import NFSTREAM_OPTIONS
    frozen = json.loads((ROOT/'configs/nfstream.json').read_text())
    if frozen['package_version'] != '6.6.0' or frozen['streamer'] != NFSTREAM_OPTIONS:
        raise ValueError('USTC frozen NFStream configurations disagree.')
    return dict(NFSTREAM_OPTIONS)
