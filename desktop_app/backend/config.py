import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PG_PROJECT_ROOT', APP.parent)).resolve()
RUNTIME = APP / 'runtime'
RUNTIME.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
research_packages = ROOT / '.venv' / 'Lib' / 'site-packages'
if research_packages.is_dir():
    sys.path.append(str(research_packages))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
for variable in ('HF_HOME','TRANSFORMERS_CACHE','SENTENCE_TRANSFORMERS_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR','TEMP','TMP'):
    directory = RUNTIME / 'cache' / variable.lower()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
DATASETS = ('USTC-TFC2016', 'CIC-Darknet2020')
PCAP_UNAVAILABLE = ('PCAP inference is unavailable for this model because the validated '
                    'feature-extraction contract is not available.')
