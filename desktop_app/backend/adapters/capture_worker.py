"""Isolated, spawn-safe NFStream host. Uses the original frozen factory/options."""
import csv
import sys
from backend.config import APP
from backend.utils.write_guard import install
from backend.utils.paths import output_path
from backend.adapters.contracts import extraction_options
from src.data.ustc_prepare_dataset import load_streamer

install(APP)

def main():
    source, destination = sys.argv[1:]
    factory = load_streamer()
    with output_path(destination).open('x',encoding='utf-8',newline='') as stream:
        writer = None
        for flow in factory(source=source,**extraction_options()):
            record = dict(zip(flow.keys(),flow.values()))
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)

if __name__=='__main__':
    main()
