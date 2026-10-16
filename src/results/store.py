from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from src.common.io import write_csv_atomic, write_json_atomic

def save_evaluation(out: Path, identity: Mapping[str, Any], evaluation: Mapping[str, Any], class_ids: Sequence[int]):
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    out.mkdir(parents=True)
    overall = [{**identity, "metric": k, "value": "" if v is None else v} for k,v in evaluation["overall"].items()]
    write_csv_atomic(out/"metrics_overall.csv", [*identity.keys(),"metric","value"], overall)
    write_csv_atomic(out/"metrics_per_class.csv",
                     [*identity.keys(),"class_id","class_name","precision","recall","f1","support"],
                     [{**identity,**r} for r in evaluation["per_class"]])
    write_csv_atomic(out/"confusion_matrix_long.csv",
                     [*identity.keys(),"actual_class_id","predicted_class_id","count"],
                     [{**identity,**r} for r in evaluation["confusion_matrix_long"]])
    write_csv_atomic(out/"calibration_bins.csv",
                     [*identity.keys(),"bin_id","lower","upper","count","accuracy","mean_confidence","absolute_gap"],
                     [{**identity,**r} for r in evaluation["calibration_bins"]])
    pred_fields = [*identity.keys(),"sample_index","y_true","y_pred","confidence","correct"] + [f"p_class_{c}" for c in class_ids]
    write_csv_atomic(out/"predictions.csv", pred_fields, [{**identity,**r} for r in evaluation["predictions"]])
    write_json_atomic(out/"summary.json", {"identity":dict(identity),"overall":evaluation["overall"],"class_ids":list(class_ids)})
