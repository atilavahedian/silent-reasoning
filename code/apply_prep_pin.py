"""Put the prep VM's pin into configs/data_manifest.json (registration A2 item 1a, A3.1: the pin row's evidence).

    python3 code/apply_prep_pin.py [--pin results/prep/pin.json] [--env configs/reference_env.json] [--dry-run]

Reads pin.json (written by the prep job: {"model": {id, revision, revision_requested, files, weights_sha256}, "python",
"versions", "time"}) and the VM's reference_env.json, and writes, additively:
- models[id == pin.model.id].revision / weights_sha256 / revision_status / pinned_at / files_sha256;
- a "gpu_environment" block (the VM's python, torch, CUDA, cuDNN, driver, transformers, GPU) with its source;
- mac_environment.status = "not used in the 72-hour window (PLAN.md A2: GPU only)".
Nothing else in the manifest changes (the data hashes stay as the critic signed them). Prints the before/after lines
for the critic's row. Ours.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ENV_KEYS = ("python", "platform", "torch", "cuda", "cudnn", "driver", "nvidia_driver", "transformers", "numpy", "scipy",
            "gpu", "gpu_name", "gpu_type", "device", "hostname", "image", "machine")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", default=str(EXP / "results" / "prep" / "pin.json"))
    ap.add_argument("--env", default=str(EXP / "configs" / "reference_env.json"))
    ap.add_argument("--manifest", default=str(EXP / "configs" / "data_manifest.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    pin = json.loads(Path(a.pin).read_text())
    model = pin.get("model") or {}
    if not (model.get("revision") and model.get("weights_sha256")):
        print(f"REFUSED: {a.pin} has no model.revision + model.weights_sha256: {json.dumps(model)[:300]}")
        return 1
    man = json.loads(Path(a.manifest).read_text())
    hit = [m for m in man.get("models", []) if m.get("id") == model.get("id")]
    if not hit:
        print(f"REFUSED: manifest has no models entry for {model.get('id')!r}")
        return 1
    m = hit[0]
    before = {k: m.get(k) for k in ("revision", "weights_sha256", "revision_status")}
    m["revision"] = model["revision"]
    m["weights_sha256"] = model["weights_sha256"]
    m["revision_requested"] = model.get("revision_requested", "main")
    m["weights_file"] = model.get("weights_file", "model.safetensors")
    m["files_sha256"] = model.get("files", {})
    m["pinned_at"] = pin.get("time")
    m["revision_status"] = (f"pinned by the prep VM at {pin.get('time')}: hub revision {model['revision']} resolved from "
                            f"{model.get('revision_requested', 'main')}; model.safetensors sha256 {model['weights_sha256']}; "
                            f"every later VM fetches this revision by commit and the launcher refuses a different hash")
    env = {}
    if Path(a.env).exists():
        raw = json.loads(Path(a.env).read_text())
        env = {k: raw[k] for k in ENV_KEYS if k in raw}
        if not env:        # unknown layout: keep it whole, it is small
            env = raw
    man["gpu_environment"] = {**env, "source": f"{Path(a.env).relative_to(EXP) if Path(a.env).is_relative_to(EXP) else a.env}"
                              f" (the prep VM's env.json; registration A3 item 3); pin.json python {pin.get('python')}",
                              "versions_from_pin": pin.get("versions")}
    mac = man.setdefault("mac_environment", {})
    mac["status"] = "not used in the 72-hour window (PLAN.md A2: GPU only)"
    print("before:", json.dumps(before))
    print("after: ", json.dumps({k: m[k] for k in ("revision", "weights_sha256")}))
    print("gpu_environment:", json.dumps({k: v for k, v in man["gpu_environment"].items() if k != "versions_from_pin"})[:400])
    if a.dry_run:
        print("dry run: manifest not written")
        return 0
    Path(a.manifest).write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
