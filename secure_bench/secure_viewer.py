
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

"""Analysis UI for Secure/Baseline/CaMeL runs. Flask app over results/ (paths.RESULTS).

Each run dir is tagged by WEBSITE (omnizon, udriver, …) and RUN TYPE (Secure | Baseline | Attack), parsed
from its name. The index lists every run with filter chips (website × run type) and official + corrected
verdicts. A run page renders per its type:
  - Secure   : per-turn ACTION / RESULT / REASONING / MEMORY-K / trusted V⊤ + untrusted V_g screens.
  - Baseline : per-step model reasoning+action + the screenshot the agent saw (plain CUA, no masking).
Every run exports to PDF at /run/<name>.pdf; /all.zip bundles the current filter's PDFs.

Run:  bash secure_viewer.sh    Env: VIEWER_PORT (8899), VIEWER_ROOT (alternate results dir).
"""
import ast
import glob
import html
import io
import json
import os
import paths
import re
import subprocess
import sys
import zipfile

from flask import Flask, Response, abort, send_file

BR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VIEWER_ROOT") or paths.RESULTS
sys.path.insert(0, os.path.join(BR, "eval_fixed"))
import corrected_eval as CE  # noqa: E402
import report_pdf as RP  # noqa: E402

app = Flask(__name__)
OP_COLOR = {"query": "#e8590c", "compute": "#2f9e44", "act": "#1971c2",
            "return": "#9c36b5", "error": "#e03131"}
RT_COLOR = {"Secure": "#2f9e44", "NoQuery": "#f08c00", "Baseline": "#1971c2", "Attack": "#e03131", "Camel-CUA": "#7048e8"}


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def classify(name):
    """dir name -> (website, run_type, task). Handles BOTH tag orderings that exist in results/:
    secure = <task>_v2_<model>_secure ; camel = <task>_v2_camel_<model> ; baseline = <task>_v2[_<model>].
    Strip the model tag FIRST (it can sit before OR after the run-type tag) so run-type detection is
    uniform. The actual model is shown per-run from meta.json; Opus/Claude runs carry no tag."""
    b = name
    for mt in ("_gemini", "_gpt", "_opus"):
        if mt in b:
            b = b.replace(mt, "", 1)
            break
    if b.endswith("_secure"):
        rt, b = "Secure", b[:-len("_secure")]
    elif b.endswith("_noquery"):
        rt, b = "NoQuery", b[:-len("_noquery")]
    elif b.endswith("_attack_miss"):
        rt, b = "Attack", b[:-len("_attack_miss")]
    elif b.endswith("_attack"):
        rt, b = "Attack", b[:-len("_attack")]
    elif b.endswith("_camel"):
        rt, b = "Camel-CUA", b[:-len("_camel")]
    elif b.endswith("_baseline"):
        rt, b = "Baseline", b[:-len("_baseline")]
    else:
        rt = "Baseline"
    website = b.split("-", 1)[0]
    task = re.sub(r"_v\d+$", "", b)
    return website, rt, task


_SEED_RE = re.compile(r"^seed(\d+)$")


def _decode(name):
    """Run identifier -> (dir, classify_name, seed). Multi-seed runs use a flat '<parent>~seed<k>' id (URL-safe,
    single URL segment) that maps to <parent>/seed<k>; flat runs pass through. Used by the /run, /img, PDF routes."""
    if "~seed" in name:
        parent, sk = name.split("~seed", 1)
        return os.path.join(ROOT, parent, f"seed{sk}"), parent, (int(sk) if sk.isdigit() else None)
    return os.path.join(ROOT, name), name, None


def model_tag(m):
    """meta.json model id -> a friendly filter label (the vision/action model). For CaMeL runs meta.model
    is the action VLM; for baseline/secure it's the model. Planner (always GPT-5) is separate."""
    s = (m or "").lower()
    if "gemini" in s or "google" in s:
        return "Gemini"
    if "gpt" in s or "openai" in s or s.startswith(("o1", "o3", "o4")):
        return "GPT-5"
    if "claude" in s or "opus" in s or "sonnet" in s or "anthropic" in s:
        return "Opus"
    if "tars" in s:
        return "UI-TARS"
    if "opencua" in s:
        return "OpenCUA"
    return m or "?"


def _is_run(d):
    return os.path.isdir(d) and (os.path.exists(os.path.join(d, "result.json"))
                                 or os.path.exists(os.path.join(d, "meta.json")))


def _run_units():
    """Yield (run_dir, identifier, classify_name, seed); descends into seed<k>/ subdirs (multi-seed layout)."""
    for d0 in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(d0):
            continue
        parent = os.path.basename(d0)
        seed_dirs = [s for s in sorted(glob.glob(os.path.join(d0, "seed*")))
                     if _SEED_RE.match(os.path.basename(s)) and _is_run(s)]
        if seed_dirs and not _is_run(d0):
            for sd in seed_dirs:
                k = int(_SEED_RE.match(os.path.basename(sd)).group(1))
                yield sd, f"{parent}~seed{k}", parent, k
        elif _is_run(d0):
            yield d0, parent, parent, None


def _runs():
    out = []
    for d, ident, cname, seed in _run_units():
        website, rt, task = classify(cname)
        if rt == "Attack":          # benign utility comparison only (Baseline / Secure / CaMeL-CUA)
            continue
        meta = _load(os.path.join(d, "meta.json"), {}) or {}
        res = _load(os.path.join(d, "result.json"), {}) or {}
        es = _load(os.path.join(d, "env_state.json"), {})
        secure = os.path.exists(os.path.join(d, "op_trace.json"))
        if secure:
            steps = res.get("turns")
        else:
            steps = len(glob.glob(os.path.join(d, "step_*.png"))) or None
        corrected = items = None
        if es:
            try:
                corrected = CE.evaluate(task, es) == "SUCCESS"
                items = "; ".join(f"{i['id']}:{i['name'][:30]} x{i['qty']}" for i in CE.order_items(es)) or "(no order)"
            except Exception:
                pass
        out.append({"name": ident, "task": task, "website": website, "run_type": rt, "seed": seed,
                    "goal": meta.get("goal", ""), "model": meta.get("model", "?"), "steps": steps,
                    "official": res.get("official_passed", res.get("passed", res.get("webarena_passed"))),
                    "corrected": corrected, "items": items or "-"})
    # order: website, then run_type, then task-number, then seed
    def keyf(r):
        m = re.search(r"-(\d+)$", r["task"])
        return (r["website"], r["run_type"], int(m.group(1)) if m else 0, r.get("seed") if r.get("seed") is not None else -1)
    return sorted(out, key=keyf)


def _badge(v):
    if v is True:
        return '<span class="v pass">PASS</span>'
    if v is False:
        return '<span class="v fail">FAIL</span>'
    return '<span class="v na">—</span>'


def _rt_tag(rt):
    return f'<span class=tag style="background:{RT_COLOR.get(rt, "#868e96")}">{html.escape(rt)}</span>'


CSS = """
*{box-sizing:border-box}
body{background:#0e1013;color:#e7e9ec;font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,sans-serif;margin:0;padding:24px 28px;max-width:1180px}
a{color:#74c0fc;text-decoration:none} a:hover{text-decoration:underline}
h1{font-size:24px;margin:2px 0 6px} h2{font-size:16px;margin:0}
.mut{color:#8b929c} .row{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
table.idx{border-collapse:collapse;width:100%;margin-top:14px}
table.idx th,table.idx td{padding:9px 11px;border-bottom:1px solid #262b33;text-align:left;vertical-align:top}
table.idx th{color:#9aa1ab;font-weight:600;font-size:13px}
.v{font-weight:700;font-size:12px;padding:1px 9px;border-radius:11px}
.v.pass{color:#0e1013;background:#40c057} .v.fail{color:#fff;background:#e03131} .v.na{color:#8b929c;background:#20242b}
.tag{display:inline-block;color:#fff;font-weight:700;font-size:11px;padding:1px 8px;border-radius:6px}
.btn{display:inline-block;padding:7px 14px;border-radius:8px;background:#1971c2;color:#fff;font-weight:600;font-size:13px}
.btn:hover{background:#1c7ed6;text-decoration:none}
.filters{margin:14px 0 4px;display:flex;flex-direction:column;gap:8px}
.filters .grp{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.filters .glbl{color:#7d8894;font-size:12px;width:64px;text-transform:uppercase;letter-spacing:.5px}
.chip{cursor:pointer;font-size:13px;padding:3px 12px;border-radius:14px;border:1px solid #333a45;color:#c9d2dc;background:#161b23;user-select:none}
.chip.on{background:#1971c2;border-color:#1971c2;color:#fff;font-weight:600}
.hdr{background:#151922;border:1px solid #262b33;border-radius:12px;padding:16px 18px;margin:10px 0 18px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
.turn{border:1px solid #262b33;border-radius:12px;margin:16px 0;background:#12151b;overflow:hidden}
.turn-top{display:flex;align-items:center;gap:12px;padding:12px 16px;background:#161b23;border-bottom:1px solid #262b33}
.chipn{font-weight:700;font-size:12px;letter-spacing:.4px;color:#fff;padding:3px 11px;border-radius:20px}
.turn-body{padding:14px 16px}
.sec{margin:12px 0} .sec:first-child{margin-top:0}
.lbl{font-size:11px;letter-spacing:.6px;text-transform:uppercase;color:#7d8894;font-weight:700;margin-bottom:5px}
.action{font-size:15px;color:#e7e9ec} .action b{color:#fff}
.json{background:#0a0c10;border:1px solid #262b33;border-radius:7px;padding:8px 11px;font-family:ui-monospace,SFMono-Regular,monospace;font-size:12.5px;white-space:pre-wrap;overflow-x:auto;color:#c9d2dc}
.result{color:#ffd8a8;font-size:13.5px}
.reason{background:#0f1620;border:1px solid #223047;border-left:3px solid #4c7bd0;border-radius:7px;padding:10px 13px;white-space:pre-wrap;font-size:13.5px;line-height:1.6;color:#d5dbe3;max-height:520px;overflow:auto}
table.k{border-collapse:collapse;margin-top:2px;font-size:12.5px}
table.k th,table.k td{border:1px solid #2b323c;padding:4px 10px;text-align:left}
table.k th{background:#1b212b;color:#aeb6c1;font-weight:600}
table.k td.m{color:#8b929c;font-family:ui-monospace,monospace}
table.k td.val{font-family:ui-monospace,monospace;color:#8ce99a}
.shots{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px}
.shot{flex:1 1 520px;min-width:320px} .vg{flex:0 0 220px}
.cap{color:#7d8894;font-size:12px;margin:3px 0}
img{max-width:100%;border:1px solid #262b33;border-radius:7px;background:#000;display:block}
"""

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style>{body}"""


@app.route("/")
def index():
    runs = _runs()
    websites = sorted({r["website"] for r in runs})
    rtypes = sorted({r["run_type"] for r in runs}, key=lambda x: list(RT_COLOR).index(x) if x in RT_COLOR else 9)
    _MORDER = ["Opus", "Gemini", "GPT-5", "UI-TARS", "OpenCUA"]
    models = sorted({model_tag(r["model"]) for r in runs},
                    key=lambda x: _MORDER.index(x) if x in _MORDER else 9)
    rows = []
    for r in runs:
        rows.append(
            f'<tr data-ws="{html.escape(r["website"])}" data-rt="{html.escape(r["run_type"])}" data-mdl="{html.escape(model_tag(r["model"]))}">'
            f'<td><a href="/run/{r["name"]}"><b>{html.escape(r["task"])}</b></a>'
            f'<div class=mut style="font-size:12px">{html.escape(str(r["goal"])[:70])}</div></td>'
            f'<td class=mut>{html.escape(r["website"])}</td>'
            f'<td>{_rt_tag(r["run_type"])}</td>'
            f'<td class=mut style="font-size:12px">{html.escape(str(r["model"]))}</td>'
            f'<td>{r["steps"] if r["steps"] is not None else "—"}</td>'
            f'<td>{_badge(r["official"])}</td><td>{_badge(r["corrected"])}</td>'
            f'<td class=mut style="font-size:12px">{html.escape(str(r["items"])[:80])}</td>'
            f'<td><a class=btn href="/run/{r["name"]}.pdf">PDF</a></td></tr>')

    def chips(field, vals):
        cs = [f'<span class="chip on" data-f="{field}" data-v="all">all</span>']
        cs += [f'<span class=chip data-f="{field}" data-v="{html.escape(v)}">{html.escape(v)}</span>' for v in vals]
        return "".join(cs)

    filters = (f'<div class=filters><div class=grp><span class=glbl>Website</span>{chips("ws", websites)}</div>'
               f'<div class=grp><span class=glbl>Run</span>{chips("rt", rtypes)}</div>'
               f'<div class=grp><span class=glbl>Model</span>{chips("mdl", models)}</div></div>')
    script = """<script>
let fws='all',frt='all',fmdl='all';
function apply(){document.querySelectorAll('tr[data-ws]').forEach(function(r){
  var ok=(fws=='all'||r.dataset.ws==fws)&&(frt=='all'||r.dataset.rt==frt)&&(fmdl=='all'||r.dataset.mdl==fmdl);r.style.display=ok?'':'none';});
  document.querySelectorAll('.chip').forEach(function(c){
    var on=(c.dataset.f=='ws'&&c.dataset.v==fws)||(c.dataset.f=='rt'&&c.dataset.v==frt)||(c.dataset.f=='mdl'&&c.dataset.v==fmdl);c.classList.toggle('on',on);});
  var parts=[];if(fws!='all')parts.push('ws='+encodeURIComponent(fws));if(frt!='all')parts.push('rt='+encodeURIComponent(frt));if(fmdl!='all')parts.push('mdl='+encodeURIComponent(fmdl));
  var q=parts.join('&');
  var z=document.getElementById('zipall');if(z)z.href='/all.zip'+(q?'?'+q:'');}
document.querySelectorAll('.chip').forEach(function(c){c.onclick=function(){
  if(c.dataset.f=='ws')fws=c.dataset.v;else if(c.dataset.f=='rt')frt=c.dataset.v;else fmdl=c.dataset.v;apply();};});
apply();
</script>"""
    body = ('<div class=hdr style="align-items:center"><div><h1 style="margin:0">Secure-eval runs</h1>'
            "<p class=mut style='margin:4px 0 0'>Secure loop · Baseline CUA · … &nbsp;|&nbsp; "
            "official = agisdk (known SKU bugs) · corrected = catalog-derived</p></div>"
            '<div><a id=zipall class=btn href="/all.zip">⬇ Download all PDFs (zip)</a></div></div>'
            + filters +
            "<table class=idx><tr><th>task</th><th>website</th><th>run</th><th>model</th><th>steps</th>"
            "<th>official</th><th>corrected</th><th>ordered items</th><th></th></tr>"
            + "".join(rows) + "</table>" + script)
    return PAGE.format(title="Secure-eval runs", css=CSS, body=body)


def _k_table_html(kdict):
    header, rows = RP.k_grid(kdict)
    if not header:
        return '<span class=mut>K is empty</span>'
    h = "".join(f"<th>{html.escape(c)}</th>" for c in header)
    trs = ""
    for row in rows:
        trs += "<tr>" + f'<td class=m>{html.escape(row[0])}</td>' + "".join(
            f'<td class=val>{html.escape(c)}</td>' for c in row[1:]) + "</tr>"
    return f"<table class=k><tr>{h}</tr>{trs}</table>"


def _g_table_html(gdict):
    """G payload handles → a table showing the BYTES (researcher view; the actuator never saw these —
    it only got handle names + counts). None if G is empty (so closed-only runs show nothing)."""
    rows = RP.g_grid(gdict)
    if not rows:
        return None
    trs = ""
    for name_, source, n, vals in rows:
        vlist = " · ".join(f'<span class=val>{html.escape(v)}</span>' for v in vals) or '<span class=mut>(empty)</span>'
        src = f' <span class=mut>from {html.escape(source)}</span>' if source else ""
        trs += f'<tr><td class=m>{html.escape(name_)}{src} <span class=mut>({n})</span></td><td>{vlist}</td></tr>'
    return ('<table class=k><tr><th>handle</th>'
            '<th>payload bytes — researcher view only; the actuator never sees these</th></tr>'
            f'{trs}</table>')


def _render_secure(d, name):
    trace = _load(os.path.join(d, "op_trace.json"), []) or []

    def img(fn):
        return f'/img/{name}/{fn}'

    blocks = []
    for e in trace:
        t = e.get("turn")
        op = e.get("op", {}) or {}
        kind = op.get("op", "?")
        fb = e.get("feedback", "") or ""
        color = OP_COLOR.get(kind, "#495057")
        rawp = os.path.join(d, f"turn_{t}_raw.txt")
        reason = open(rawp, encoding="utf-8", errors="replace").read().strip() if os.path.exists(rawp) else ""
        secs = [f'<div class=sec><div class=lbl>Action</div>'
                f'<div class=action>{html.escape(RP.op_summary(op))}</div>'
                f'<div class=json style="margin-top:6px">{html.escape(json.dumps(op, ensure_ascii=False))}</div></div>']
        if fb:
            secs.append(f'<div class=sec><div class=lbl>Result</div><div class=result>{html.escape(fb)}</div></div>')
        if kind == "return":
            mrp = os.path.join(d, "model_response.txt")
            ans = open(mrp, encoding="utf-8", errors="replace").read().strip() if os.path.exists(mrp) else ""
            if ans:
                secs.append('<div class=sec><div class=lbl>Answer delivered to user '
                            '(materialized — payload handles substituted at the user sink)</div>'
                            f'<div class=result>{html.escape(ans)}</div></div>')
        if reason:
            secs.append(f'<div class=sec><div class=lbl>Reasoning</div><div class=reason>{html.escape(reason)}</div></div>')
        _kdict = RP.k_after(d, t)
        if _kdict:                                          # noquery has no K store — skip the empty table
            secs.append(f'<div class=sec><div class=lbl>Memory K — after this turn</div>{_k_table_html(_kdict)}</div>')
        _gtab = _g_table_html(RP.g_after(d, t))
        if _gtab:
            secs.append(f'<div class=sec><div class=lbl>Memory G — payload handles (after this turn)</div>{_gtab}</div>')
        shots = ""
        if os.path.exists(os.path.join(d, f"turn_{t}_vtop.png")):
            shots += (f'<div class=shot><div class=cap>TRUSTED V⊤ — what the actuator saw</div>'
                      f'<img src="{img(f"turn_{t}_vtop.png")}"></div>')
        if kind == "query":
            outc = dict(re.findall(r"#(\d+):(\w+)", fb))
            for p in sorted(glob.glob(os.path.join(d, f"turn_{t}_mark*_vg.png")),
                            key=lambda p: int(re.search(r"mark(\d+)_", p).group(1))):
                m = re.search(r"mark(\d+)_", p).group(1)
                shots += (f'<div class="shot vg"><div class=cap>UNTRUSTED V_g — mark #{m} → '
                          f'<b>{html.escape(outc.get(m,"?"))}</b></div><img src="{img(os.path.basename(p))}"></div>')
        if kind == "act" and os.path.exists(os.path.join(d, f"turn_{t}_after.png")):
            shots += (f'<div class=shot><div class=cap>result after act</div>'
                      f'<img src="{img(f"turn_{t}_after.png")}"></div>')
        if shots:
            secs.append(f'<div class=sec><div class=lbl>Screens</div><div class=shots>{shots}</div></div>')
        blocks.append(
            f'<div class=turn><div class=turn-top><span class=chipn style="background:{color}">{kind.upper()}</span>'
            f'<h2>Turn {t}</h2></div><div class=turn-body>' + "".join(secs) + '</div></div>')
    return "".join(blocks)


def _render_baseline(d, name):
    traj = _load(os.path.join(d, "trajectory.json"), {}) or {}
    steps = traj.get("steps", []) or []

    def img(fn):
        return f'/img/{name}/{fn}'

    blocks = []
    for s in steps:
        i = s.get("step")
        resp = (s.get("model_response") or "").strip()
        acts = s.get("parsed_actions") or []
        secs = []
        if resp:
            secs.append(f'<div class=sec><div class=lbl>Reasoning + action</div>'
                        f'<div class=reason>{html.escape(resp)}</div></div>')
        if acts:
            secs.append(f'<div class=sec><div class=lbl>Parsed action</div>'
                        f'<div class=json>{html.escape("; ".join(acts))}</div></div>')
        obs = os.path.join(d, f"step_{i-1}.png")     # the screen the agent saw before this step
        if os.path.exists(obs):
            secs.append(f'<div class=sec><div class=lbl>Screen the agent saw</div>'
                        f'<div class=shots><div class=shot><img src="{img(f"step_{i-1}.png")}"></div></div></div>')
        blocks.append(
            f'<div class=turn><div class=turn-top><span class=chipn style="background:#1971c2">STEP</span>'
            f'<h2>Step {i}</h2></div><div class=turn-body>' + "".join(secs) + '</div></div>')
    # final screen
    finals = sorted(glob.glob(os.path.join(d, "step_*.png")), key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if finals:
        blocks.append(f'<div class=turn><div class=turn-body><div class=lbl>Final screen</div>'
                      f'<div class=shots><div class=shot><img src="{img(os.path.basename(finals[-1]))}"></div></div></div></div>')
    return "".join(blocks)


# ---------------------------------------------------------------------------
#  CaMeL-CUA renderer: the GPT-5 plan as a control-flow graph (executed path
#  highlighted), the plan's own "# >>>" step outcomes, and the physical-action
#  timeline with screenshots — all from what's in the run dir.
# ---------------------------------------------------------------------------
_CAMEL_TOOLS = {"find", "find_element_by_text", "left_single", "left_double", "right_single",
    "left_triple", "middle_click", "left_press", "type_text", "scroll", "hotkey", "press", "key",
    "hold_key", "drag", "hover", "verify_hypothesis", "summarize_screenshot_content", "get_page_text",
    "get_page_elements", "check_done", "query_ai_assistant", "mark_done", "mark_fail", "wait", "no_op",
    "run_single_claude"}


def _camel_executed_texts(d):
    """Stripped code-line texts rendered green (=executed) in plan_highlighted.html."""
    p = os.path.join(d, "plan_highlighted.html")
    if not os.path.exists(p):
        return set()
    t = open(p, encoding="utf-8", errors="ignore").read()
    return {html.unescape(m).strip() for m in re.findall(r"color:\s*green'?>(.*?)</span>", t, re.S)
            if html.unescape(m).strip()}


def _camel_call(stmt):
    v = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in _CAMEL_TOOLS:
        return v.func.id, v
    return None


_CAMEL_ACTUATORS = {"left_single", "left_double", "right_single", "left_triple", "middle_click",
    "type_text", "scroll", "hotkey", "press", "key", "hold_key", "drag", "left_press",
    "left_mouse_down", "left_mouse_up"}


def _camel_exec_steps(d):
    """Executed tool-call plan statements, in program order: {line, tool, first, src, act}."""
    p = os.path.join(d, "plan.txt")
    code = open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
    if m:
        code = m.group(1)
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    lines = code.split("\n")
    executed = _camel_executed_texts(d)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.Expr)):
            c = _camel_call(node)
            if c and 1 <= node.lineno <= len(lines):
                first = lines[node.lineno - 1].strip()
                if first in executed:
                    try:
                        full = ast.get_source_segment(code, node) or first
                    except Exception:
                        full = first
                    out.append({"line": node.lineno, "tool": c[0], "first": first,
                                "src": full, "act": c[0] in _CAMEL_ACTUATORS})
    out.sort(key=lambda s: s["line"])
    return out


def _camel_traces_by_src(d):
    """Map a plan code-line (stripped) -> the '# >>>' outputs the plan printed right after it."""
    p = os.path.join(d, "plan_annotated.txt")
    if not os.path.exists(p):
        return {}
    res, last = {}, None
    for ln in open(p, encoding="utf-8", errors="ignore"):
        st = ln.strip()
        if st.startswith("# >>>"):
            if last is not None:
                res.setdefault(last, []).append(st[5:].strip())
        elif st and not st.startswith("#"):
            last = st
    return res


_CAMEL_CSS = ("<style>.pl{font-family:monospace;font-size:12.5px;background:#0e1218;border:1px solid #232a33;"
    "border-left:3px solid #7048e8;border-radius:6px;padding:8px 10px;color:#d7f5e0;overflow-x:auto;"
    "white-space:pre-wrap}.why{font-size:12.5px;color:#e6a817;margin-top:2px;padding:5px 9px;"
    "border-left:2px solid #e6a817;background:#1a1508;border-radius:4px;white-space:pre-wrap}"
    ".trace{font-size:13px;color:#c9d2dc;padding:5px 0;border-bottom:1px solid #1c222b;display:flex;gap:8px}"
    ".tn{color:#9775fa;font-weight:700;min-width:22px}"
    ".planhl{font-family:monospace;font-size:12px;line-height:1.5;background:#0e1218;border:1px solid #232a33;"
    "border-radius:8px;padding:12px;overflow:auto;max-height:560px}.ln{color:#7d8894;font-size:11px}</style>")


def _camel_action(e):
    a = str(e.get("args", ""))
    m = re.search(r"pyautogui\.[A-Za-z_]+\([^\)]*\)", a) or re.search(r"\bWAIT\b", a)
    if m:
        return m.group(0)
    f = str(e.get("function", "")).replace("\\n", " ").replace("\n", " ").strip()
    return f or html.unescape(a)[:120]


def _camel_full_plan(d):
    ph = os.path.join(d, "plan_highlighted.html")
    if not os.path.exists(ph):
        return ""
    return ('<div class=turn><div class=turn-body><div class=lbl>Complete plan (GPT-5) &mdash; '
            'executed lines in green</div><div class=planhl>'
            + open(ph, encoding="utf-8", errors="ignore").read() + '</div></div></div>')


def _render_camel(d, name):
    def img(fn):
        return f"/img/{name}/{fn}"
    sl = os.path.join(d, "step_log.jsonl")
    if os.path.exists(sl):
        steps = []
        for ln in open(sl, encoding="utf-8", errors="ignore"):
            try:
                steps.append(json.loads(ln))
            except Exception:
                pass
        parts = [_CAMEL_CSS, '<div class=turn><div class=turn-body><div class=lbl>Execution &mdash; each plan '
                 f'line as it ran ({len(steps)} tool-calls), with its result / reasoning and any screenshot.'
                 '</div></div></div>']
        for s in steps:
            tool = str(s.get("tool", ""))
            secs = [f'<div class=sec><div class=lbl><span class=ln>plan line {s.get("line")}</span> &middot; '
                    f'<b>{html.escape(tool)}</b></div><div class=pl>{html.escape(str(s.get("src", "")))}</div></div>']
            res = str(s.get("result", "")).strip()
            if res and res not in ("None", "NoneType"):
                lab = "printed" if tool == "print" else "result"
                secs.append(f'<div class=sec><div class=lbl>{lab}</div><div class=why>{html.escape(res[:400])}</div></div>')
            for sh in (s.get("shots") or []):
                secs.append('<div class=sec><div class=lbl>screen after</div><div class=shots>'
                            f'<div class=shot><img src="{img(sh)}"></div></div></div>')
            parts.append(f'<div class=turn><div class=turn-top><span class=chipn style="background:#7048e8">'
                         f'STEP {s.get("seq")}</span></div><div class=turn-body>{"".join(secs)}</div></div>')
        parts.append(_camel_full_plan(d))
        return "".join(parts)
    # ---- fallback (runs without step logging): physical actions + the plan's progress notes ----
    traj = _load(os.path.join(d, "traj.json"), []) or []
    ann = os.path.join(d, "plan_annotated.txt")
    notes = ([ln.strip()[5:].strip() for ln in open(ann, encoding="utf-8", errors="ignore")
              if ln.strip().startswith("# >>>")] if os.path.exists(ann) else [])
    shots = sorted(glob.glob(os.path.join(d, "step_*.png")),
                   key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    parts = [_CAMEL_CSS, '<div class=turn><div class=turn-body><div class=lbl>(No per-step log for this run '
             "&mdash; re-run with step logging for line-by-line detail.) Physical actions + the plan's own "
             'progress notes below.</div></div></div>']
    if notes:
        nn = "".join(f'<div class=trace><span class=tn>{i+1}</span><span>{html.escape(t)}</span></div>'
                     for i, t in enumerate(notes))
        parts.append("<div class=turn><div class=turn-body><div class=lbl>Plan's own progress notes</div>"
                     + nn + "</div></div>")
    for k, e in enumerate(traj):
        stt = str(e.get("result_status", ""))
        col = "#2f9e44" if stt == "OK" else "#e6a817"
        sec = [f'<div class=sec><div class=lbl>action</div><div class=pl>{html.escape(_camel_action(e))}</div>'
               f'<div style="margin-top:5px;font-size:12px">&rarr; <b style="color:{col}">{html.escape(stt)}</b></div></div>']
        if k < len(shots):
            sec.append('<div class=sec><div class=lbl>screen after</div><div class=shots>'
                       f'<div class=shot><img src="{img(os.path.basename(shots[k]))}"></div></div></div>')
        parts.append(f'<div class=turn><div class=turn-top><span class=chipn style="background:#7048e8">'
                     f'STEP {k+1}</span></div><div class=turn-body>{"".join(sec)}</div></div>')
    parts.append(_camel_full_plan(d))
    return "".join(parts)


@app.route("/run/<name>")
def run(name):
    d, cname, seed = _decode(name)
    if ".." in name or not os.path.isdir(d):
        abort(404)
    website, rt, task = classify(cname)
    meta = _load(os.path.join(d, "meta.json"), {}) or {}
    res = _load(os.path.join(d, "result.json"), {}) or {}
    try:
        ordered, corrected = RP.order_line(d)
    except Exception:
        ordered, corrected = None, None
    off = res.get("official_passed", res.get("passed"))
    wa = res.get("webarena_verdict")                       # WebArena runs (reddit): a single verdict
    if wa is not None and off is None:
        off = res.get("webarena_passed")
    verdict_html = (f'<div>webarena {_badge(off)} <span class=mut>{html.escape(str(wa))}</span></div>'
                    if wa is not None
                    else f'<div>official {_badge(off)}</div><div>corrected {_badge(corrected)}</div>')
    secure = os.path.exists(os.path.join(d, "op_trace.json"))
    if rt == "Camel-CUA":
        body_runs = _render_camel(d, name)
    elif secure:
        body_runs = _render_secure(d, name)
    else:
        body_runs = _render_baseline(d, name)
    steps = res.get("turns") if secure else len(glob.glob(os.path.join(d, "step_*.png")))
    header = (
        f'<p><a href="/">&larr; all runs</a></p>'
        f'<div class=hdr><div><h1 style="margin:0">{html.escape(task)} {_rt_tag(rt)}</h1>'
        f'<div class=mut style="margin:4px 0">{html.escape(str(meta.get("goal","")))}</div>'
        f'<div class=row style="margin-top:8px">{verdict_html}'
        f'<div class=mut>website: {html.escape(website)}</div>'
        f'<div class=mut>model: {html.escape(str(meta.get("model","?")))}</div>'
        f'<div class=mut>steps: {steps}</div></div>'
        f'<div class=mut style="margin-top:6px">ordered: {html.escape(ordered)}</div></div>'
        f'<div><a class=btn href="/run/{name}.pdf">⬇ Download PDF</a></div></div>')
    return PAGE.format(title=task, css=CSS, body=header + body_runs)


@app.route("/run/<name>.pdf")
def run_pdf(name):
    d, _cname, _seed = _decode(name)
    if ".." in name or not os.path.isdir(d):
        abort(404)
    try:
        pdf = RP.build_run_pdf(d)
    except Exception as e:
        return Response(f"PDF build failed: {e}", status=500, mimetype="text/plain")
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{name}.pdf"'})


@app.route("/all.zip")
def all_zip():
    from flask import request
    ws = request.args.get("ws")
    rt = request.args.get("rt")
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in _runs():
            if (ws and r["website"] != ws) or (rt and r["run_type"] != rt):
                continue
            d, _c, _s = _decode(r["name"])
            try:
                zf.writestr(f"{r['name']}.pdf", RP.build_run_pdf(d))
                n += 1
            except Exception as e:
                zf.writestr(f"{r['name']}.ERROR.txt", f"PDF build failed: {e}")
    if not n:
        return Response("no runs to export", status=404, mimetype="text/plain")
    buf.seek(0)
    fn = "secure_runs_pdfs" + (f"_{ws}" if ws else "") + (f"_{rt}" if rt else "")
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fn}.zip"'})


@app.route("/img/<name>/<path:fn>")
def image(name, fn):
    if ".." in name or ".." in fn:
        abort(404)
    d, _c, _s = _decode(name)
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        abort(404)
    return send_file(p)


if __name__ == "__main__":
    port = int(os.environ.get("VIEWER_PORT", "8899"))
    print(f"[viewer] http://0.0.0.0:{port}  root={ROOT}  (SSH-tunnel this port)")
    app.run(host="0.0.0.0", port=port, debug=False)
