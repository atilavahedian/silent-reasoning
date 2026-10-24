#!/usr/bin/env python3
"""Download and preprocess the three Coconut tasks into <repo>/data/ (gitignored) and write a SHA-256 manifest.

Usage (from anywhere; nothing here needs torch):
    python3 code/fetch_data.py \
        [--data-dir data] [--only prosqa gsm prontoqa] [--prontoqa-dir DIR] [--force]

* ProsQA: data/prosqa_{train,valid,test}.json from facebookresearch/coconut, the `main` ref resolved to a commit at
  fetch time and downloaded from that commit (the files are used as released, no conversion).
* GSM8k-Aug: the iCoT files at da03/Internalize_CoT_Step_by_Step@e06a32ee5e4cd117171daeb4755d2a97ece62761
  (train.txt via Git LFS media URL, 88 MB), kept under data/raw/, converted with the semantics of Coconut
  preprocessing/gsm_icot.py into data/gsm_{train,valid,test}.json.
* ProntoQA: asaparov/prontoqa@f0145b867b3c106285ec9ea1941a3f6eb7c6162d cloned into its own directory (untrusted
  data: run with `python3 -I` from a separate working directory; its own folder is appended to sys.path only so
  that its sibling modules import, after the standard library), the Coconut README generator command, then the
  semantics of Coconut preprocessing/prontoqa.py into data/prontoqa_{train,valid,test}.json (9000 / 200 / rest).

Every downloaded source and every output gets {sha256, bytes, records} in
configs/data_manifest.json, with URLs, commits and the generator command.
The manifest is printed at the end.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
ROOT = HERE.parents[3]
MANIFEST = EXP / "configs" / "data_manifest.json"

COCONUT_REPO = "https://github.com/facebookresearch/coconut"
PROSQA_URL = "https://raw.githubusercontent.com/facebookresearch/coconut/{ref}/data/prosqa_{split}.json"
ICOT_COMMIT = "e06a32ee5e4cd117171daeb4755d2a97ece62761"
GSM_URLS = {
    "train": f"https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/{ICOT_COMMIT}/data/gsm8k/train.txt",
    "valid": f"https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/{ICOT_COMMIT}/data/gsm8k/valid.txt",
    "test": f"https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/{ICOT_COMMIT}/data/gsm8k/test.txt",
}
PRONTOQA_REPO = "https://github.com/asaparov/prontoqa"
PRONTOQA_COMMIT = "f0145b867b3c106285ec9ea1941a3f6eb7c6162d"
PRONTOQA_ARGS = ["--model-name", "json", "--model-size", "dummy", "--ordering", "random", "--num-trials", "10000",
                 "--few-shot-examples", "0", "--ontology", "fictional", "--min-hops", "5", "--max-hops", "5",
                 "--hops-skip", "1"]
PRONTOQA_RAW = "5hop_0shot_random.json"

# Runs the untrusted generator: isolated interpreter (-I: no cwd/script dir/user site on sys.path), its folder
# appended (not prepended) so the standard library cannot be shadowed, argv as the README command.
_RUNNER = (
    "import sys, runpy\n"
    "src = sys.argv[1]\n"
    "sys.path.append(src)\n"
    "sys.argv = ['run_experiment.py'] + sys.argv[2:]\n"
    "runpy.run_path(src + '/run_experiment.py', run_name='__main__')\n"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_records(path: Path) -> int:
    if path.suffix == ".json":
        with open(path) as f:
            obj = json.load(f)
        return len(obj)
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def entry(path: Path, role: str, **extra) -> dict:
    return {"path": os.path.relpath(path, ROOT), "role": role, "sha256": sha256(path), "bytes": path.stat().st_size,
            "records": count_records(path), **extra}


def download(url: str, dest: Path, force: bool, tries: int = 4) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest} exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f, 1 << 20)
            if expected is not None and tmp.stat().st_size != int(expected):
                raise IOError(f"short read {tmp.stat().st_size} of {expected}")
            tmp.replace(dest)
            print(f"[get] {url} -> {dest} ({dest.stat().st_size} bytes)")
            return
        except Exception as e:  # network hiccup: retry with backoff
            print(f"[retry {attempt}/{tries}] {url}: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"download failed: {url}")


def resolve_ref(repo: str, ref: str) -> str:
    out = subprocess.run(["git", "ls-remote", repo, ref], check=True, capture_output=True, text=True).stdout.split()
    if not out:
        raise RuntimeError(f"cannot resolve {ref} in {repo}")
    return out[0]


def fetch_prosqa(data: Path, force: bool, ref: str) -> dict:
    commit = resolve_ref(COCONUT_REPO, ref) if len(ref) != 40 else ref
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = []
    for split in ("train", "valid", "test"):
        url = PROSQA_URL.format(ref=commit, split=split)
        dest = data / f"prosqa_{split}.json"
        download(url, dest, force)
        files.append(entry(dest, "source+output", url=url,
                           url_as_requested=PROSQA_URL.format(ref=ref, split=split)))
    return {"status": "fetched", "repo": COCONUT_REPO, "ref_requested": ref, "commit": commit,
            "commit_resolution": f"`git ls-remote {COCONUT_REPO} {ref}` returned {commit} at {when}; the three files "
                                 "were downloaded from raw.githubusercontent.com URLs pinned to that commit (`url`), "
                                 "not from the moving `main` URL (`url_as_requested`)",
            "files": files}


def convert_gsm(src: Path, dest: Path) -> None:
    """Coconut preprocessing/gsm_icot.py main(split), verbatim semantics (question unstripped, steps split on
    single spaces, answer after the last '##')."""
    with open(src) as f:
        data = f.readlines()
    data = [
        {
            "question": d.split("||")[0],
            "steps": d.split("||")[1].split("##")[0].strip().split(" "),
            "answer": d.split("##")[-1].strip(),
        }
        for d in data
    ]
    with open(dest, "w") as f:
        json.dump(data, f)


def fetch_gsm(data: Path, force: bool) -> dict:
    files = []
    for split, url in GSM_URLS.items():
        raw = data / "raw" / f"gsm_{split}.txt"
        download(url, raw, force)
        files.append(entry(raw, "source", url=url))
        out = data / f"gsm_{split}.json"
        convert_gsm(raw, out)
        files.append(entry(out, "output", derived_from=os.path.relpath(raw, ROOT),
                           converter="Coconut preprocessing/gsm_icot.py semantics"))
    return {"status": "fetched", "repo": "https://github.com/da03/Internalize_CoT_Step_by_Step",
            "commit": ICOT_COMMIT, "files": files}


def convert_prontoqa(raw: Path, data: Path) -> list[Path]:
    """Coconut preprocessing/prontoqa.py, verbatim semantics: question + " " + query; steps = chain_of_thought
    sentences joined in pairs; answer; splits [:9000], [9000:9200], [9200:]."""
    with open(raw) as f:
        file = json.load(f)
    out = []
    for k, v in file.items():
        example = v["test_example"]
        out.append({
            "question": example["question"] + " " + example["query"],
            "steps": [" ".join(example["chain_of_thought"][i: i + 2])
                      for i in range(0, len(example["chain_of_thought"]), 2)],
            "answer": example["answer"],
        })
    paths = []
    for name, part in (("train", out[:9000]), ("valid", out[9000:9200]), ("test", out[9200:])):
        p = data / f"prontoqa_{name}.json"
        with open(p, "w") as f:
            json.dump(part, f)
        paths.append(p)
    return paths


def _trim_log(path: Path, keep: int = 200_000) -> None:
    """The generator prints hundreds of MB of rejection-sampling warnings; keep the line count and the tail."""
    if not path.exists() or path.stat().st_size <= keep:
        return
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)
        f.seek(-keep, os.SEEK_END)
        tail = f.read()
    path.write_bytes(f"[trimmed by fetch_data.py: {n_lines} lines, last {keep} bytes kept]\n".encode() + tail)


def _run_generator(src: Path, work: Path, force: bool) -> Path:
    raw = work / PRONTOQA_RAW
    if raw.exists() and not force:
        print(f"[skip] {raw} exists")
        return raw
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copy(src / "bad_patterns.txt", work / "bad_patterns.txt")  # read from the cwd by run_experiment.py
    t0 = time.time()
    with open(work / "generator.log", "w") as log:
        r = subprocess.run([sys.executable, "-I", "-c", _RUNNER, str(src), *PRONTOQA_ARGS], cwd=work,
                           stdout=log, stderr=subprocess.STDOUT)
    _trim_log(work / "generator.log")
    if r.returncode != 0 or not raw.exists() or raw.stat().st_size == 0:
        tail = (work / "generator.log").read_text()[-2000:]
        raise RuntimeError(f"ProntoQA generator failed (exit {r.returncode}):\n{tail}")
    print(f"[gen] ProntoQA generated in {time.time() - t0:.0f} s -> {raw}")
    return raw


def fetch_prontoqa(data: Path, src: Path, force: bool, runs: int = 2) -> dict:
    """Clone the pinned generator, run the README command `runs` times (each in its own working directory), keep the
    first run's output, record every run's hash (registration D9: generator nondeterminism)."""
    cmd = "python run_experiment.py " + " ".join(PRONTOQA_ARGS)
    if src.exists() and any(src.iterdir()):
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        if head.strip() != PRONTOQA_COMMIT:
            raise RuntimeError(f"{src} is not an empty dir nor a prontoqa checkout at {PRONTOQA_COMMIT}")
    else:
        src.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", PRONTOQA_REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "-c", "advice.detachedHead=false", "checkout", "-q", PRONTOQA_COMMIT],
                       check=True)
    raws = [_run_generator(src, data / "raw" / ("prontoqa_run" if k == 1 else f"prontoqa_run{k}"), force)
            for k in range(1, runs + 1)]
    gen_runs = [{"run": k, **entry(r, "source" if k == 1 else "source (second run, hash only; not used)",
                                   generated_by=cmd)} for k, r in enumerate(raws, 1)]
    files = [{"path": "<prontoqa checkout outside the repo>/run_experiment.py", "role": "generator",
              "sha256": sha256(src / "run_experiment.py"), "bytes": (src / "run_experiment.py").stat().st_size,
              "git_commit": PRONTOQA_COMMIT}, gen_runs[0]]
    for p in convert_prontoqa(raws[0], data):
        files.append(entry(p, "output", derived_from=os.path.relpath(raws[0], ROOT),
                           converter="Coconut preprocessing/prontoqa.py semantics", label="regenerated data"))
    hashes = [g["sha256"] for g in gen_runs]
    return {"status": "fetched", "label": "regenerated data", "repo": PRONTOQA_REPO, "commit": PRONTOQA_COMMIT,
            "generator_command": cmd,
            "generator_seed": "none on the command line; run_experiment.py calls random.seed(args.seed) and "
                              "np.random.seed(args.seed) with the argparse default --seed 62471893",
            "generator_python": sys.version.split()[0],
            "generator_invocation": "python3 -I -c <runpy runner> <checkout> <args>, cwd = data/raw/prontoqa_run*/ "
                                    "(holds a copy of bad_patterns.txt); checkout appended to sys.path",
            "generation_runs": gen_runs, "generation_runs_identical": len(set(hashes)) == 1,
            "kept_run": 1, "files": files,
            "note": "not byte-comparable with the paper's ProntoQA files (never released); every ProntoQA cell is "
                    "labelled regenerated data"}


MODELS = [  # registration 1.1: HF revisions are resolved and pinned on the Mac (huggingface.co is blocked here)
    {"id": "openai-community/gpt2", "size": "124M", "n_ne": 85_056_000},
    {"id": "openai-community/gpt2-medium", "size": "355M", "n_ne": 302_311_424},
    {"id": "openai-community/gpt2-large", "size": "774M", "n_ne": 708_390_400},
]


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": sys.platform}
    for mod in ("torch", "transformers", "numpy", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # recorded, not fatal: fetching needs none of them except scipy for ProntoQA
            env[mod] = f"unavailable ({type(e).__name__})"
    return env


def overlap_block(data: Path) -> dict:
    """13-gram overlap (latent.contamination.overlap_report) of train vs valid and train vs test for every task
    whose files exist: primary = test question only, secondary = question + steps + answer (registration A1)."""
    sys.path.insert(0, str(ROOT / "common"))
    from latent.contamination import overlap_report
    out = {}
    for task in ("prosqa", "prontoqa", "gsm"):
        paths = {s: data / f"{task}_{s}.json" for s in ("train", "valid", "test")}
        if not all(p.exists() for p in paths.values()):
            out[task] = {"status": "data absent"}
            continue
        rep = overlap_report(str(paths["train"]), {s: str(paths[s]) for s in ("valid", "test")})
        rel = lambda x: os.path.relpath(x, ROOT)
        out[task] = {"train_file": rel(rep["train_file"]), "train_fields": rep["train_fields"], "n": rep["n"]}
        for split, r in rep["splits"].items():
            out[task][split] = {
                kind: {k: r[kind][k] for k in ("test_items", "test_items_with_overlap", "fraction",
                                               "test_ngrams", "test_ngrams_shared", "test_fields")}
                for kind in ("primary", "secondary")}
        print(f"[13-gram] {task}: test primary {out[task]['test']['primary']['fraction']:.4f}, "
              f"secondary {out[task]['test']['secondary']['fraction']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--only", nargs="*", default=["prosqa", "gsm", "prontoqa"])
    ap.add_argument("--prontoqa-dir", default=os.environ.get("PRONTOQA_DIR", str(ROOT / "scratch" / "prontoqa")),
                    help="empty directory for the ProntoQA clone (untrusted code; kept out of the repo)")
    ap.add_argument("--prontoqa-runs", type=int, default=2, help="generator runs; all hashed, the first is kept")
    ap.add_argument("--coconut-ref", default="main")
    ap.add_argument("--force", action="store_true", help="re-download and regenerate even if files exist")
    ap.add_argument("--skip-overlap", action="store_true", help="do not recompute the 13-gram overlap block")
    a = ap.parse_args()
    data = Path(a.data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    manifest.setdefault("tasks", {})
    manifest["data_dir"] = os.path.relpath(data, ROOT)
    manifest["script"] = os.path.relpath(HERE, ROOT)
    manifest["generated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["local_date"] = datetime.date.today().isoformat()
    manifest["build_environment"] = environment()
    manifest["mac_environment"] = {"python": None, "torch": None, "transformers": None,
                                   "status": "to be filled on the Mac (registration 1.4)"}
    manifest["models"] = [{**m, "revision": None, "weights_sha256": None,
                           "revision_status": "placeholder: resolve and pin on the Mac before any training run"}
                          for m in MODELS]

    steps = {"prosqa": lambda: fetch_prosqa(data, a.force, a.coconut_ref),
             "gsm": lambda: fetch_gsm(data, a.force),
             "prontoqa": lambda: fetch_prontoqa(data, Path(a.prontoqa_dir).resolve(), a.force, a.prontoqa_runs)}
    failed = False
    for name in a.only:
        try:
            manifest["tasks"][name] = steps[name]()
        except Exception as e:
            failed = True
            manifest["tasks"][name] = {"status": "not fetched here", "reason": f"{type(e).__name__}: {e}"[:1500]}
            print(f"[fail] {name}: {e}")
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    if not a.skip_overlap:
        manifest["contamination_13gram"] = {
            "module": "common/latent/contamination.py",
            "definition": "fraction of eval items sharing >= 1 word 13-gram (lowercased [a-z0-9]+ words, spans "
                          "inside one record) with any train record's question + steps",
            "tasks": overlap_block(data)}
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
