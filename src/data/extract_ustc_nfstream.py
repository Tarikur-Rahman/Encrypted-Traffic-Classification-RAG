from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from nfstream import NFStreamer
from src.common.hashing import sha256_file
from src.common.io import write_json_atomic

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = args.project_root.resolve()
    cfg = json.loads((root / "configs/nfstream.json").read_text(encoding="utf-8"))
    source_root = root / "data/interim/USTC-TFC2016/pcaps"
    pcaps = sorted(list(source_root.rglob("*.pcap")) + list(source_root.rglob("*.pcapng")))
    if not pcaps:
        raise SystemExit(f"No prepared USTC PCAPs under {source_root}")
    out_root = root / "data/interim/USTC-TFC2016/nfstream_native"
    print(f"PCAPS={len(pcaps)}")
    print(json.dumps(cfg, indent=2))
    if not args.execute:
        print("PREFLIGHT_ONLY=True")
        return 0
    if out_root.exists() and any(out_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite populated NFStream directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    s = cfg["streamer"]
    for idx, pcap in enumerate(pcaps):
        rel = pcap.relative_to(source_root)
        category = rel.parts[0]
        family = rel.parts[1] if len(rel.parts) > 1 else pcap.stem
        dst = out_root / category / family / f"{pcap.stem}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        streamer = NFStreamer(
            source=str(pcap),
            decode_tunnels=s["decode_tunnels"],
            snapshot_length=s["snapshot_length"],
            idle_timeout=s["idle_timeout"],
            active_timeout=s["active_timeout"],
            accounting_mode=s["accounting_mode"],
            n_dissections=s["n_dissections"],
            statistical_analysis=s["statistical_analysis"],
            splt_analysis=s["splt_analysis"],
            n_meters=s["n_meters"],
            max_nflows=s["max_nflows"],
            performance_report=s["performance_report"],
            system_visibility_mode=s["system_visibility_mode"],
        )
        flow_count = streamer.to_csv(path=str(dst), columns_to_anonymize=[], flows_per_file=0, rotate_files=0)
        manifest.append({
            "source_pcap": str(rel).replace("\\", "/"),
            "source_sha256": sha256_file(pcap),
            "category": category,
            "family": family,
            "output_csv": str(dst.relative_to(root)).replace("\\", "/"),
            "output_sha256": sha256_file(dst),
            "flow_count": int(flow_count),
        })
        print(f"[{idx+1}/{len(pcaps)}] {rel} -> {flow_count} flows")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json_atomic(
        root / "metadata/manifests" / f"ustc_nfstream_extraction_{stamp}.json",
        {
            "status": "PASS",
            "nfstream_config": cfg,
            "pcap_count": len(pcaps),
            "total_flows": sum(x["flow_count"] for x in manifest),
            "artifacts": manifest,
        }
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
