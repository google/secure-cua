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

"""Read-only cross-arm token / wall-time / USD collation for the instrumented final campaign.

Walks  <ROOT>/<model>_<mode>/<task>_<mode>/result.json , sums the per-task result["metrics"]
(written by run_secure_webarena.py / run_baseline_webarena.py) per component per arm across all
tasks, and emits, under ROOT:
  - metrics_summary.csv   one row per (arm, component) + a per-arm TOTAL row.
  - metrics_summary.json  same numbers machine-readable, plus per-task rows.

Components: "actuator" + "shadow" for secure arms; "baseline" for baseline arms.

USD is an ESTIMATE from the EDITABLE PRICES table below (published list prices as of the run —
CONFIRM / EDIT to your negotiated Vertex / Azure rates). Cost per component:
    cost = input/1e6 * price_in  +  billable_out/1e6 * price_out
where billable_out = output + thinking ONLY for Gemini (thoughts_token_count is billed on top of
candidates). For Anthropic, thinking is already folded into output_tokens; for OpenAI, reasoning
tokens are a SUBSET of completion_tokens — so for both, billable_out = output. These asymmetries are
applied here at cost time; the stored per-task counts stay raw/uninterpreted.

Usage:  python collate_metrics.py [ROOT]        (ROOT default: ./vm_results_final)
"""
import csv
import glob
import json
import os
import sys

# ---- EDITABLE per-model-family pricing, USD per 1,000,000 tokens. -----------------------------
# ---- Published list prices, web-verified Jul 2026 — EDIT for your negotiated Vertex/Azure rates.
# ---- NOTE Gemini has a long-context tier: >200k-token context is billed $4/$18 (vs $2/$12 below),
# ---- so the secure Gemini actuator (full history often >200k) is an UNDER-estimate at these rates.
PRICES = {
    "opus":   {"in":  5.00, "out": 25.00},   # Anthropic Claude Opus 4.8 (list, Jul 2026)
    "gemini": {"in":  2.00, "out": 12.00},   # Google Gemini 3.1 Pro — STANDARD tier <=200k ctx (>200k = $4/$18)
    "gpt":    {"in":  1.25, "out": 10.00},   # Azure OpenAI GPT-5 (list, Jul 2026)
}

COMPONENTS = ("actuator", "shadow", "baseline")
_KEYS = ("calls", "input", "output", "thinking", "time")


def _family(name):
    n = (name or "").lower()
    if "opus" in n or "claude" in n:
        return "opus"
    if "gemini" in n:
        return "gemini"
    if "gpt" in n or n.startswith(("o1", "o3", "o4")):
        return "gpt"
    return None


def _cost(family, inp, out, think):
    p = PRICES.get(family)
    if not p:
        return None
    billable_out = out + (think if family == "gemini" else 0)
    return round(inp / 1e6 * p["in"] + billable_out / 1e6 * p["out"], 4)


def _model_of(arm_dir):
    """Best-effort model id from any task's meta.json (else None -> fall back to arm-dir name)."""
    for mj in sorted(glob.glob(os.path.join(arm_dir, "*", "meta.json"))):
        try:
            return json.load(open(mj, encoding="utf-8")).get("model")
        except Exception:
            continue
    return None


def collate_arm(arm_dir):
    arm = os.path.basename(arm_dir.rstrip("/"))
    model = _model_of(arm_dir)
    family = _family(model) or _family(arm)
    agg = {c: {k: 0 for k in _KEYS} for c in COMPONENTS}
    tasks, task_time_sum = [], 0.0
    for rj in sorted(glob.glob(os.path.join(arm_dir, "*", "result.json"))):
        try:
            res = json.load(open(rj, encoding="utf-8"))
        except Exception:
            continue
        m = res.get("metrics")
        if not m:
            continue
        row = {"task": res.get("task") or os.path.basename(os.path.dirname(rj)),
               "verdict": res.get("webarena_verdict"), "task_time": m.get("task_time")}
        task_time_sum += float(m.get("task_time") or 0)
        for c in COMPONENTS:
            cm = m.get(c)
            if not cm:
                continue
            for k in _KEYS:
                agg[c][k] += cm.get(k, 0) or 0
            row[c] = cm
        tasks.append(row)
    n = len(tasks)
    rows = []
    for c in COMPONENTS:
        a = agg[c]
        if not (a["calls"] or a["input"] or a["output"]):
            continue                                       # component unused by this arm
        billable_out = a["output"] + (a["thinking"] if family == "gemini" else 0)
        rows.append({
            "arm": arm, "model": model or family, "component": c, "n_tasks": n,
            "calls": a["calls"], "input": a["input"], "output": a["output"],
            "thinking": a["thinking"], "total_tokens": a["input"] + billable_out,
            "time_s": round(a["time"], 1), "time_s_per_task": round(a["time"] / n, 1) if n else 0,
            "usd": _cost(family, a["input"], a["output"], a["thinking"]),
        })
    arm_usd = round(sum(r["usd"] for r in rows if r["usd"] is not None), 4) if rows else None
    return {"arm": arm, "model": model, "family": family, "n_tasks": n,
            "task_time_total_s": round(task_time_sum, 1),
            "task_time_mean_s": round(task_time_sum / n, 1) if n else 0,
            "arm_usd": arm_usd, "components": rows, "tasks": tasks}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vm_results_final")
    if not os.path.isdir(root):
        sys.exit(f"no such dir: {root}")
    arm_dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                      if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "result.json")))
    if not arm_dirs:
        sys.exit(f"no arm dirs with */result.json under {root}")
    arms = [collate_arm(d) for d in arm_dirs]

    csv_path = os.path.join(root, "metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "model", "component", "n_tasks", "calls", "input_tokens", "output_tokens",
                    "thinking_tokens", "total_tokens", "time_s", "time_s_per_task", "est_usd"])
        for a in arms:
            for r in a["components"]:
                w.writerow([r["arm"], r["model"], r["component"], r["n_tasks"], r["calls"], r["input"],
                            r["output"], r["thinking"], r["total_tokens"], r["time_s"],
                            r["time_s_per_task"], r["usd"]])
            w.writerow([a["arm"], a["model"] or a["family"], "TOTAL", a["n_tasks"], "", "", "", "", "",
                        a["task_time_total_s"], a["task_time_mean_s"], a["arm_usd"]])

    json_path = os.path.join(root, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"root": root, "prices": PRICES, "arms": arms}, f, indent=2, default=str)

    print("=" * 92)
    print("USD is an ESTIMATE from an EDITABLE price table in collate_metrics.py (PRICES).")
    print("Confirm / adjust to your negotiated Vertex / Azure rates before quoting any cost.")
    print("=" * 92)
    hdr = f"{'arm':<20}{'component':<11}{'calls':>7}{'in_tok':>13}{'out_tok':>13}{'think':>11}{'time_s':>10}{'usd$':>10}"
    print(hdr)
    for a in arms:
        for r in a["components"]:
            print(f"{r['arm']:<20}{r['component']:<11}{r['calls']:>7}{r['input']:>13}{r['output']:>13}"
                  f"{r['thinking']:>11}{r['time_s']:>10.1f}{(r['usd'] or 0):>10.3f}")
        print(f"{a['arm']:<20}{'TOTAL':<11}{'':>7}{'':>13}{'':>13}{'':>11}"
              f"{a['task_time_total_s']:>10.1f}{(a['arm_usd'] or 0):>10.3f}")
    print(f"\nwrote {csv_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
