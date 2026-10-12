#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pass@k reporter for the multi-seed runs.

Every arm runs each task across K independent seeds (repeats). This tool reads all the per-seed result.json
files under a results root, groups them by (arm, task), and reports the UNBIASED pass@k estimator
(Chen et al. 2021): for a task with n samples of which c succeed,

    pass@k = 1 - C(n-c, k) / C(n, k)          (= 1 when n-c < k)

averaged over tasks. It also reports solve-rate (fraction of tasks solved by >=1 seed) and the empirical
pass@1 (mean c/n). Works across BOTH verdict schemas (WebArena `webarena_verdict`, omnizon/realbench
`corrected_verdict` / `passed`) and BOTH layouts (flat `<task>_<arm>/result.json` = 1 seed, or
`<task>_<arm>/seed<k>/result.json` = K seeds).

  python secure_bench/passk.py results/runs                 # per-arm pass@1..K table
  python secure_bench/passk.py results/runs --by-site       # + per-site breakdown
  python secure_bench/passk.py results/runs --csv out.csv   # also write a CSV

`pass_at_k` and `passed` are importable (run_suite.py reuses them).
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------- core metric + verdict ------------------
def passed(res):
    """True iff this result.json represents a SUCCESS, across both verdict schemas."""
    return (res.get("webarena_verdict") == "SUCCESS"
            or res.get("corrected_verdict") == "SUCCESS"
            or res.get("passed") is True
            or res.get("official_passed") is True and res.get("corrected_verdict") is None)


def has_verdict(res):
    """True iff the run produced a real (non-error) verdict — SUCCESS or FAILURE in either schema."""
    return (res.get("webarena_verdict") in ("SUCCESS", "FAILURE")
            or res.get("corrected_verdict") in ("SUCCESS", "FAILURE")
            or isinstance(res.get("passed"), bool))


def pass_at_k(n, c, k):
    """Unbiased pass@k for a task with n samples, c correct. Returns None if k > n (can't estimate)."""
    if k > n:
        return None
    if n - c < k:                       # every way to pick k of the n samples includes a success
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------- discovery ------------------------------
_SEED_RE = re.compile(r"^seed(\d+)$")
_MODES = ("secure", "noquery", "baseline", "camel")          # order matters: check longer/specific first
_MODEL_TAGS = ("gemini", "gpt", "opus")


def _site_of(task):
    return task.rsplit("-", 1)[0] if task and "-" in task else (task or "?")


def _classify(run_name, parent_name, meta, res):
    """(arm, model, site, task) for a run directory. Prefer meta.json; fall back to the dir name.
    run_name = the arm/task dir (seed<k> already stripped); parent_name = its parent (qtag for CaMeL)."""
    task = (meta.get("task") if meta else None) or res.get("task") or run_name
    site = _site_of(task)

    # CaMeL: run dir is "<task>_camel" (webarena) or "<task>_v2_camel_<qtag>" (realbench); qtag identifies the arm.
    if "_camel" in run_name:
        qtag = None
        m = re.search(r"_camel_(q\w+)$", run_name)          # realbench: ..._camel_qopus
        if m:
            qtag = m.group(1)
        elif parent_name and parent_name.startswith("q"):    # webarena: camel_webarena/<qtag>/<task>_camel
            qtag = parent_name
        return (f"camel/{qtag}" if qtag else "camel", qtag or "camel", site, task)

    # masking arms: mode from meta, else from the dir suffix.
    mode = (meta.get("mode") if meta else None)
    if not mode:
        for m in _MODES:
            if run_name.endswith("_" + m) or ("_" + m + "_") in run_name or ("_" + m + "s") in run_name:
                mode = m
                break
        mode = mode or "baseline"
    model = (meta.get("model") if meta else None)
    if not model:
        model = next((t for t in _MODEL_TAGS if t in run_name), "opus")
    fam = "gemini" if "gemini" in (model or "") else "gpt" if any(x in (model or "") for x in ("gpt", "o1", "o3", "o4")) else "opus"
    return (f"{mode}/{fam}", model, site, task)


def collect(root):
    """Walk `root`, return groups: {(arm, site, task): [is_success_bool, ...]} plus an error tally."""
    groups = defaultdict(list)
    errors = defaultdict(int)                                # (arm, site, task) -> # non-verdict samples
    seen = set()
    for rp in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
        rp = os.path.abspath(rp)
        if rp in seen:
            continue
        seen.add(rp)
        d = os.path.dirname(rp)
        seed_m = _SEED_RE.match(os.path.basename(d))
        run_dir = os.path.dirname(d) if seed_m else d        # strip a seed<k> level
        run_name = os.path.basename(run_dir)
        parent_name = os.path.basename(os.path.dirname(run_dir))
        try:
            res = json.load(open(rp, encoding="utf-8"))
        except Exception:
            continue
        meta = None
        mp = os.path.join(os.path.dirname(rp), "meta.json")
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp, encoding="utf-8"))
            except Exception:
                meta = None
        arm, _model, site, task = _classify(run_name, parent_name, meta, res)
        key = (arm, site, task)
        if has_verdict(res):
            groups[key].append(bool(passed(res)))
        else:
            errors[key] += 1                                 # ERROR/TIMEOUT/partial: counts toward n as a failure
            groups[key].append(False)
    return groups, errors


# ---------------------------------------------------------------- aggregation + report -------------------
def _summarize(task_nc, kmax):
    """task_nc = list of (n, c). Return {'tasks','samples','solved','solve_rate','emp_pass1','pass@k':{k:..}}."""
    tasks = len(task_nc)
    samples = sum(n for n, _ in task_nc)
    solved = sum(1 for _, c in task_nc if c >= 1)
    emp1 = (sum(c / n for n, c in task_nc if n) / tasks) if tasks else None
    passk = {}
    for k in range(1, kmax + 1):
        vals = [pass_at_k(n, c, k) for n, c in task_nc]
        vals = [v for v in vals if v is not None]            # tasks with n>=k
        passk[k] = (sum(vals) / len(vals)) if vals else None
    return {"tasks": tasks, "samples": samples, "solved": solved,
            "solve_rate": (solved / tasks) if tasks else None, "emp_pass1": emp1, "passk": passk}


def report(groups, errors, by_site=False):
    # per (arm[, site]) -> list of (n,c)
    per_arm = defaultdict(list)
    per_arm_site = defaultdict(list)
    kmax = 1
    for (arm, site, task), verds in groups.items():
        n, c = len(verds), sum(verds)
        kmax = max(kmax, n)
        per_arm[arm].append((n, c))
        per_arm_site[(arm, site)].append((n, c))
    out = {"kmax": kmax, "arms": {}}
    for arm in sorted(per_arm):
        s = _summarize(per_arm[arm], kmax)
        s["n_errors"] = sum(errors[k] for k in errors if k[0] == arm)
        if by_site:
            s["by_site"] = {site: _summarize(per_arm_site[(arm, site)], kmax)
                            for (a, site) in sorted(per_arm_site) if a == arm}
        out["arms"][arm] = s
    return out


def _fmt(v):
    return "  -  " if v is None else f"{v:5.1%}"


def print_table(rep, by_site=False):
    kmax = rep["kmax"]
    ks = list(range(1, kmax + 1))
    hdr = f"{'arm':22} {'tasks':>5} {'smpl':>5} {'solve':>6} " + " ".join(f"p@{k:<3}" for k in ks) + f" {'err':>4}"
    print(hdr); print("-" * len(hdr))
    for arm, s in rep["arms"].items():
        row = (f"{arm:22} {s['tasks']:>5} {s['samples']:>5} {_fmt(s['solve_rate']):>6} "
               + " ".join(_fmt(s['passk'][k]) for k in ks) + f" {s['n_errors']:>4}")
        print(row)
        if by_site and s.get("by_site"):
            for site, ss in s["by_site"].items():
                print(f"  {site:20} {ss['tasks']:>5} {ss['samples']:>5} {_fmt(ss['solve_rate']):>6} "
                      + " ".join(_fmt(ss['passk'][k]) for k in ks))
    print("\n(p@k = unbiased pass@k over seeds; solve = frac tasks solved by >=1 seed; err = non-verdict samples)")


def write_csv(rep, path):
    ks = list(range(1, rep["kmax"] + 1))
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "site", "tasks", "samples", "solved", "solve_rate", "emp_pass1"] + [f"pass@{k}" for k in ks] + ["n_errors"])
        for arm, s in rep["arms"].items():
            w.writerow([arm, "ALL", s["tasks"], s["samples"], s["solved"], s["solve_rate"], s["emp_pass1"]]
                       + [s["passk"][k] for k in ks] + [s["n_errors"]])
            for site, ss in (s.get("by_site") or {}).items():
                w.writerow([arm, site, ss["tasks"], ss["samples"], ss["solved"], ss["solve_rate"], ss["emp_pass1"]]
                           + [ss["passk"][k] for k in ks] + [""])
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="pass@k over multi-seed runs")
    ap.add_argument("root", help="results dir to scan (e.g. results/runs)")
    ap.add_argument("--by-site", action="store_true", help="also break down per site")
    ap.add_argument("--csv", metavar="PATH", help="also write a CSV")
    ap.add_argument("--json", metavar="PATH", help="also write the raw report JSON")
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f"not a directory: {a.root}")
    groups, errors = collect(a.root)
    if not groups:
        sys.exit(f"no result.json found under {a.root}")
    rep = report(groups, errors, by_site=a.by_site)
    print_table(rep, by_site=a.by_site)
    if a.csv:
        write_csv(rep, a.csv)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=2, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
