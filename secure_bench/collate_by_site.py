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

"""Per-SITE cost + time breakdown for the final campaign — actuator vs shadow shown SEPARATELY.

Reads vm_results_final/<model>_<mode>/<task>_<mode>/result.json, groups by (model, mode, site),
sums each component's tokens+time, and applies the same editable USD price table + billable-thinking
semantics as collate_metrics.py (imported, not duplicated). Site is inferred from the task id
(reddit / shopping / gitlab / shopping_admin — note shopping_admin uses an underscore, so rsplit on
the LAST '-' keeps it intact). Emits metrics_by_site.{csv,json} under ROOT + a console table.

Usage:  python collate_by_site.py [ROOT]     (ROOT default: ./vm_results_final)
"""
import csv, glob, json, os, sys, collections
import collate_metrics as CM        # reuse PRICES, _family, _cost, _model_of

SITES = ["reddit", "shopping", "gitlab", "shopping_admin"]
COMPONENTS = ("actuator", "shadow", "baseline")
KEYS = ("calls", "input", "output", "thinking", "time")


def site_of(task):
    return task.rsplit("-", 1)[0] if task else "?"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vm_results_final")
    if not os.path.isdir(root):
        sys.exit(f"no such dir: {root}")
    agg = collections.defaultdict(lambda: {k: 0 for k in KEYS})      # (model,mode,site,comp) -> sums
    ntasks = collections.defaultdict(int)                            # (model,mode,site) -> #tasks
    verd = collections.defaultdict(dict)                             # (model,mode,site) -> {task: success bool}
    fam = {}
    for arm_dir in sorted(glob.glob(os.path.join(root, "*"))):
        arm = os.path.basename(arm_dir)
        if not os.path.isdir(arm_dir) or "_" not in arm:
            continue
        model, mode = arm.split("_", 1)
        fam[model] = CM._family(CM._model_of(arm_dir) or model)
        for r in glob.glob(os.path.join(arm_dir, "*", "result.json")):
            try:
                j = json.load(open(r))
            except Exception:
                continue
            s = site_of(j.get("task", ""))
            verd[(model, mode, s)][j.get("task")] = (j.get("webarena_verdict") == "SUCCESS")
            m = j.get("metrics")
            if not m:
                continue
            ntasks[(model, mode, s)] += 1
            for c in COMPONENTS:
                cm = m.get(c)
                if not cm:
                    continue
                for k in KEYS:
                    agg[(model, mode, s, c)][k] += cm.get(k, 0) or 0

    # ---- CSV / JSON (one row per model,mode,site,component) ----
    rows = []
    for (model, mode, s, c), a in sorted(agg.items()):
        usd = CM._cost(fam.get(model), a["input"], a["output"], a["thinking"])
        rows.append({"model": model, "mode": mode, "site": s, "component": c,
                     "n_tasks": ntasks[(model, mode, s)], "calls": a["calls"],
                     "input": a["input"], "output": a["output"], "thinking": a["thinking"],
                     "time_s": round(a["time"], 1), "usd": usd})
    with open(os.path.join(root, "metrics_by_site.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "mode", "site", "component", "n_tasks", "calls",
                    "input_tok", "output_tok", "thinking_tok", "time_s", "usd"])
        for r in rows:
            w.writerow([r[k] for k in ("model", "mode", "site", "component", "n_tasks",
                        "calls", "input", "output", "thinking", "time_s", "usd")])
    json.dump(rows, open(os.path.join(root, "metrics_by_site.json"), "w"), indent=2)

    # ---- per-site PASS metrics (secure ✓, baseline ✓, utility=sec-pass among base-pass) ----
    def passes(model, mode, s):
        d = verd.get((model, mode, s), {})
        return sum(d.values()), len(d)
    def util(model, s):
        sec = verd.get((model, "secure", s), {}); bas = verd.get((model, "baseline", s), {})
        bp = [t for t in bas if bas[t]]
        return sum(1 for t in bp if sec.get(t)), len(bp)
    pass_rows = []
    for model in sorted({m for (m, _, _) in verd}):
        for s in SITES:
            sp, sn = passes(model, "secure", s); bpn, bn = passes(model, "baseline", s)
            un, ud = util(model, s)
            pass_rows.append({"model": model, "site": s, "secure_pass": sp, "secure_n": sn,
                              "baseline_pass": bpn, "baseline_n": bn, "utility_num": un, "utility_den": ud})
    with open(os.path.join(root, "metrics_by_site_pass.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "site", "secure_pass", "secure_n",
                                        "baseline_pass", "baseline_n", "utility_num", "utility_den"])
        for r in pass_rows:
            w.writerow([r[k] for k in ("model", "site", "secure_pass", "secure_n",
                        "baseline_pass", "baseline_n", "utility_num", "utility_den")])

    # ---- WIDE final CSV (+xlsx) : one row per (site, model), mirroring the presented table ----
    wide_hdr = ["site", "model", "secure_pass", "secure_n", "baseline_pass", "baseline_n",
                "utility_num", "utility_den", "actuator_usd", "shadow_usd", "secure_usd", "baseline_usd",
                "actuator_min", "shadow_min", "secure_min", "baseline_min",
                "actuator_in_tok", "actuator_out_tok", "actuator_think_tok",
                "shadow_in_tok", "shadow_out_tok", "baseline_in_tok", "baseline_out_tok", "baseline_think_tok"]
    Z = {k: 0 for k in KEYS}
    wide = []
    for s in SITES:
        for model in sorted({m for (m, _, _) in verd}):
            f = fam.get(model)
            sa = agg.get((model, "secure", s, "actuator"), Z)
            ss = agg.get((model, "secure", s, "shadow"), Z)
            bb = agg.get((model, "baseline", s, "baseline"), Z)
            au = CM._cost(f, sa["input"], sa["output"], sa["thinking"]) or 0
            su = CM._cost(f, ss["input"], ss["output"], ss["thinking"]) or 0
            bu = CM._cost(f, bb["input"], bb["output"], bb["thinking"]) or 0
            sp, sn = passes(model, "secure", s); bpn, bn = passes(model, "baseline", s); un, ud = util(model, s)
            wide.append([s, model, sp, sn, bpn, bn, un, ud,
                         round(au, 2), round(su, 2), round(au + su, 2), round(bu, 2),
                         round(sa["time"] / 60, 1), round(ss["time"] / 60, 1),
                         round((sa["time"] + ss["time"]) / 60, 1), round(bb["time"] / 60, 1),
                         sa["input"], sa["output"], sa["thinking"], ss["input"], ss["output"],
                         bb["input"], bb["output"], bb["thinking"]])
    with open(os.path.join(root, "final_site_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(wide_hdr); w.writerows(wide)
    xlsx_msg = ""
    try:
        import openpyxl
        wb = openpyxl.Workbook(); wsx = wb.active; wsx.title = "site_results"
        wsx.append(wide_hdr)
        for r in wide:
            wsx.append(r)
        wb.save(os.path.join(root, "final_site_results.xlsx"))
        xlsx_msg = " + final_site_results.xlsx"
    except Exception as e:
        xlsx_msg = f" (xlsx skipped: {type(e).__name__})"

    # ---- console table: per site, per model — pass | secure act$/shd$/total$ + times | baseline ----
    models = sorted({m for (m, _, _, _) in agg})
    print("=" * 116)
    print("PER-SITE PASS + COST (USD, editable prices) + LLM TIME (min) — actuator vs shadow SEPARATE (secure)")
    print("=" * 116)
    for s in SITES:
        print(f"\n── {s} ──")
        print(f"  {'model':<8}│{'SEC✓':>6}{'BASE✓':>7}{'util':>7} │ {'act$':>9}{'shd$':>8}{'SEC$':>8} │"
              f"{'act m':>8}{'shd m':>7} ║ {'BASE$':>7}{'base m':>8}")
        for model in models:
            sa = agg.get((model, "secure", s, "actuator")); ss = agg.get((model, "secure", s, "shadow"))
            bb = agg.get((model, "baseline", s, "baseline"))
            act = (CM._cost(fam.get(model), sa["input"], sa["output"], sa["thinking"]) if sa else 0) or 0
            shd = (CM._cost(fam.get(model), ss["input"], ss["output"], ss["thinking"]) if ss else 0) or 0
            bas = (CM._cost(fam.get(model), bb["input"], bb["output"], bb["thinking"]) if bb else 0) or 0
            am = (sa["time"] / 60) if sa else 0; sm = (ss["time"] / 60) if ss else 0; bm = (bb["time"] / 60) if bb else 0
            sp, sn = passes(model, "secure", s); bpn, bn = passes(model, "baseline", s); un, ud = util(model, s)
            print(f"  {model:<8}│{f'{sp}/{sn}':>6}{f'{bpn}/{bn}':>7}{f'{un}/{ud}':>7} │ {act:>9.2f}{shd:>8.2f}{act+shd:>8.2f} │"
                  f"{am:>8.1f}{sm:>7.1f} ║ {bas:>7.2f}{bm:>8.1f}")
    print(f"\nwrote metrics_by_site.csv/.json + metrics_by_site_pass.csv + final_site_results.csv{xlsx_msg} under {root}")


if __name__ == "__main__":
    main()
