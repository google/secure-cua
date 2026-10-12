
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

"""Read-only join of the BASELINE vs SECURE vs NO-QUERY suite results — the utility/query-necessity table.

Reads vm_results/_suite/suite_{baseline,secure,noquery}_latest.json (written by run_suite.py) and prints one
row per task with all three verdicts plus the secure op-mix (the mechanism: secure carries query/compute;
baseline & no-query are act/return only). Two claims are read straight off the columns:
  - query-necessary : secure SUCCESS / no-query FAILURE (the query capability is what unlocks the task)
  - masking-cost    : baseline SUCCESS / secure FAILURE (utility the defense's masking costs vs unrestricted CUA)
Any arm whose suite json is absent shows "-" and is skipped in its count. Pure read-only join — no scoring, no runs.

   python compare_arms.py [--out-root <dir>]
"""
import argparse
import json
import os
import re


def _load(suite_dir, mode):
    """Per-task aggregate across seeds: {task: {n, c, solved, op_mix}}. Uses the suite json's `per_task`
    (n samples, c successes) when present; falls back to deriving it from `results` for old single-seed json."""
    p = os.path.join(suite_dir, f"suite_{mode}_latest.json")
    if not os.path.exists(p):
        return {}, p
    d = json.load(open(p, encoding="utf-8"))
    mix = {}
    for r in d.get("results", []):
        mix.setdefault(r["task"], r.get("op_mix"))
    pt = d.get("per_task")
    if not pt:                                            # old single-seed suite json -> derive (n,c) from results
        agg = {}
        for r in d.get("results", []):
            a = agg.setdefault(r["task"], [0, 0])
            a[0] += 1
            a[1] += (r.get("verdict") == "SUCCESS")
        pt = {t: {"n": n, "c": c} for t, (n, c) in agg.items()}
    return {t: {"n": v["n"], "c": v["c"], "solved": v["c"] >= 1, "op_mix": mix.get(t)} for t, v in pt.items()}, p


def _mix(r):
    m = r.get("op_mix") or {}
    return " ".join(f"{k}:{v}" for k, v in sorted(m.items())) or "-"


def _sortkey(t):
    m = re.search(r"(\d+)$", t)
    return (t.split("-", 1)[0], int(m.group(1)) if m else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vm_results"))
    a = ap.parse_args()
    suite_dir = os.path.join(a.out_root, "_suite")
    bl, bp = _load(suite_dir, "baseline")
    sec, sp = _load(suite_dir, "secure")
    nq, qp = _load(suite_dir, "noquery")
    if not (bl or sec or nq):
        print(f"no suite results under {suite_dir} (expected suite_{{baseline,secure,noquery}}_latest.json)")
        return

    def cell(d, t):                                       # display: successes/seeds (e.g. "3/5"), "-" if not run
        r = d.get(t)
        return f"{r['c']}/{r['n']}" if r else "-"

    def solved(d, t):                                     # >=1 seed passed
        return bool(d.get(t, {}).get("solved"))

    tasks = sorted(set(bl) | set(sec) | set(nq), key=_sortkey)
    print(f"{'task':<16}{'BASELINE':<10}{'SECURE':<10}{'NOQUERY':<10}secure op-mix   (cell = successes/seeds)")
    print("-" * 104)
    n_bl = n_sec = n_nq = n_diag = n_tax = 0
    for t in tasks:
        n_bl += solved(bl, t); n_sec += solved(sec, t); n_nq += solved(nq, t)
        diag = solved(sec, t) and (t in nq) and not solved(nq, t)   # query-necessary: secure solves, noquery ran & never did
        tax = solved(bl, t) and (t in sec) and not solved(sec, t)   # masking utility cost
        n_diag += diag
        n_tax += tax
        flag = "  <= query-necessary" if diag else ("  <= masking-cost" if tax else "")
        print(f"{t:<16}{cell(bl,t):<10}{cell(sec,t):<10}{cell(nq,t):<10}{_mix(sec.get(t, {}))}" + flag)
    tot = len(tasks)
    print("-" * 104)
    print(f"tasks solved (>=1 seed): baseline {n_bl}/{tot}  ·  secure {n_sec}/{tot}  ·  noquery {n_nq}/{tot}")
    print(f"query-necessary (secure solved / noquery not): {n_diag}   |   masking-cost (baseline solved / secure not): {n_tax}")
    print("(per-task cell = successes/seeds; for the unbiased pass@k table use: python secure_bench/passk.py <out-root>)")


if __name__ == "__main__":
    main()
