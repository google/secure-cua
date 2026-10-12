
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

"""Render one secure-loop run to a PDF (reportlab): per turn -> action, reasoning, memory (K), screens.

Also exports the small shared helpers op_summary()/k_after()/order_line() that the Flask viewer reuses so
the HTML page and the PDF stay in lock-step. build_run_pdf(run_dir) -> bytes.
"""
import glob
import io
import json
import os
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# op accent colors, tuned for legibility on white
OP_HEX = {"query": "#c2410c", "compute": "#15803d", "act": "#1d4ed8", "return": "#7c3aed", "error": "#b91c1c"}
INK, MUT, RULE, GREEN = "#1f2937", "#6b7280", "#e5e7eb", "#166534"


# ---------------------------------------------------------------- shared helpers (viewer reuses) ----
def op_summary(op):
    """One-line human-readable description of an op."""
    k = op.get("op", "?")
    if k == "query":
        t = op.get("return_type", "?")
        ev = op.get("enum_values")
        typ = f"{t}{'=' + '/'.join(map(str, ev)) if ev else ''}"
        dest = "G" if t == "payload" else "K"
        return f'QUERY  “{op.get("rho","")}”  over marks {op.get("over", [])}  →  {dest}.{op.get("bind","?")} ({typ})'
    if k == "compute":
        a = op.get("args")
        return f'COMPUTE  {op.get("operation","?")}({op.get("select","?")})' + (f'  {json.dumps(a)}' if a else "") \
               + (f'  →  {op.get("bind")}' if op.get("bind") else "")
    if k == "act":
        tg = op.get("target") or {}
        where = (f'mark #{tg["mark"]}' if "mark" in tg else
                 f'({tg.get("x")},{tg.get("y")})' if tg else "")
        extra = (f' text="{op.get("text")}"' if op.get("text") else "") \
            + (f' key={op.get("key")}' if op.get("key") else "") \
            + (f' dy={op.get("dy")}' if op.get("dy") is not None else "")
        return f'ACT  {op.get("action","?")} {where}{extra}'.rstrip()
    if k == "return":
        use = op.get("use") or []
        binds = ", ".join(f'{u.get("slot")}←{u.get("handle")}' for u in use if isinstance(u, dict))
        return f'RETURN [{op.get("status","?")}]  “{op.get("answer","")}”' + (f'  using {{{binds}}}' if binds else "")
    return json.dumps(op, ensure_ascii=False)


def k_after(run_dir, turn):
    """The K snapshot recorded after `turn` -> {col:{type,cells:{mark:val}}} (or {} if none)."""
    p = os.path.join(run_dir, f"K_turn_{turn}.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def g_after(run_dir, turn):
    """The G payload store as of `turn` -> {name:{source,entries:[...]}}. Falls back to the latest PRIOR
    snapshot when this turn's file is missing (the terminal `return` op breaks before its per-turn dump),
    so the return turn still shows the handle it materialized. NOTE: this DOES contain the opaque bytes —
    it is the researcher's artifact view, NOT the actuator's (which only saw G's schema). {} if none."""
    for t in range(int(turn), 0, -1):
        p = os.path.join(run_dir, f"G_turn_{t}.json")
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding="utf-8"))
            except Exception:
                return {}
    return {}


def g_grid(gdict):
    """G -> [(handle, source, n_entries, [values])...] for display. [] if empty."""
    rows = []
    for name, h in (gdict or {}).items():
        vals = [e.get("value", "") for e in (h.get("entries") or [])]
        rows.append((name, h.get("source", ""), len(vals), vals))
    return rows


def k_grid(kdict):
    """K -> (header list, rows list-of-lists) for tabular display. Empty -> (None, None)."""
    if not kdict:
        return None, None
    cols = list(kdict)
    marks = sorted({int(m) for c in kdict.values() for m in (c.get("cells") or {})})
    header = ["mark"] + [f"{c} ({kdict[c].get('type','?')})" for c in cols]
    rows = []
    for m in marks:
        row = [f"#{m}"]
        for c in cols:
            v = (kdict[c].get("cells") or {}).get(str(m), (kdict[c].get("cells") or {}).get(m))
            row.append("·" if v is None else str(v))
        rows.append(row)
    return header, rows


def _raw(run_dir, turn):
    p = os.path.join(run_dir, f"turn_{turn}_raw.txt")
    return open(p, encoding="utf-8", errors="replace").read().strip() if os.path.exists(p) else ""


def order_line(run_dir):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_fixed"))
        import corrected_eval as CE
        esf = os.path.join(run_dir, "env_state.json")
        if os.path.exists(esf):
            es = json.load(open(esf))
            its = CE.order_items(es)
            return "; ".join(f"{i['id']}:{i['name']} ×{i['qty']}" for i in its) or "(no order placed)", \
                   (CE.evaluate(os.path.basename(run_dir).replace("_v2_secure", ""), es) == "SUCCESS")
    except Exception:
        pass
    return "-", None


# ---------------------------------------------------------------- image embedding -------------------
def _pdf_image(path, disp_w_pt, max_px=1100):
    """Downscale a PNG (PIL) to <= max_px wide, embed as a reportlab Image at disp_w_pt wide."""
    im = PILImage.open(path).convert("RGB")
    if im.width > max_px:
        im = im.resize((max_px, round(im.height * max_px / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    disp_h = disp_w_pt * im.height / im.width
    return Image(buf, width=disp_w_pt, height=disp_h)


# ---------------------------------------------------------------- the document ----------------------
def build_run_pdf(run_dir):
    name = os.path.basename(run_dir)
    meta = json.load(open(os.path.join(run_dir, "meta.json"))) if os.path.exists(os.path.join(run_dir, "meta.json")) else {}
    res = json.load(open(os.path.join(run_dir, "result.json"))) if os.path.exists(os.path.join(run_dir, "result.json")) else {}
    trace = json.load(open(os.path.join(run_dir, "op_trace.json"))) if os.path.exists(os.path.join(run_dir, "op_trace.json")) else []
    task = meta.get("task", name.replace("_v2_secure", ""))

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5, alignment=TA_LEFT,
                          textColor=colors.HexColor(INK))
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=colors.HexColor(MUT))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, spaceAfter=3, textColor=colors.HexColor("#111827"))
    lbl = ParagraphStyle("lbl", parent=small, fontSize=7.5, textColor=colors.HexColor("#9ca3af"),
                         fontName="Helvetica-Bold", spaceBefore=9, spaceAfter=2)
    cap = ParagraphStyle("cap", parent=small, fontSize=7.5, textColor=colors.HexColor("#9ca3af"), spaceBefore=4)
    action_st = ParagraphStyle("action", parent=body, fontSize=10.5, leading=14.5)
    result_st = ParagraphStyle("result", parent=body, fontSize=9, textColor=colors.HexColor("#92400e"))
    reason_st = ParagraphStyle("reason", parent=body, fontSize=9, leading=13,
                               backColor=colors.HexColor("#f8f9fb"), borderColor=colors.HexColor("#e5e7eb"),
                               borderWidth=0.6, borderPadding=9, textColor=colors.HexColor("#374151"))
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=7.8, leading=10.5,
                          backColor=colors.HexColor("#f3f4f6"), borderColor=colors.HexColor("#e5e7eb"),
                          borderWidth=0.5, borderPadding=6, textColor=colors.HexColor("#374151"))
    th_st = ParagraphStyle("th", parent=body, fontSize=12.5, spaceBefore=2, spaceAfter=1)

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def pill(ok):
        c = "#16a34a" if ok else "#dc2626" if ok is False else "#9ca3af"
        return f"<font color='{c}'><b>{'PASS' if ok else 'FAIL' if ok is False else '—'}</b></font>"

    def turn_block(kind, t):
        c = OP_HEX.get(kind, "#4b5563")
        head = Paragraph(f"<font color='#111827'><b>Turn {t}</b></font>"
                         f"&nbsp;&nbsp;<font color='{c}'><b>{kind.upper()}</b></font>", th_st)
        rule = HRFlowable(width="100%", thickness=1.1, color=colors.HexColor(c), spaceBefore=1, spaceAfter=3)
        return [Spacer(1, 8), head, rule]

    story = []
    # --- header ---
    story.append(Paragraph(esc(task), h1))
    story.append(Paragraph(f"secure loop &nbsp;·&nbsp; {esc(meta.get('provider','?'))} tools &nbsp;·&nbsp; "
                           f"{res.get('turns','?')} turns", small))
    ordered, corrected = order_line(run_dir)
    wa = res.get("webarena_verdict")                        # WebArena/noquery runs carry a single official verdict
    story.append(Spacer(1, 3))
    if wa is not None:
        story.append(Paragraph(f"webarena&nbsp; {pill(res.get('webarena_passed'))} "
                               f"&nbsp;<font color='#6b7280'>({esc(wa)})</font>", body))
    else:
        off = res.get("official_passed", res.get("passed"))
        story.append(Paragraph(f"official&nbsp; {pill(off)} &nbsp;&nbsp;&nbsp; corrected&nbsp; {pill(corrected)}", body))
    story.append(Paragraph(f"<b>goal:</b> {esc(meta.get('goal',''))}", body))
    if wa is None:                                          # 'ordered' is an omnizon-order concept — skip for webarena
        story.append(Paragraph(f"<b>ordered:</b> <font color='{GREEN}'>{esc(ordered)}</font>", small))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.4, color=colors.HexColor("#9ca3af")))

    for e in trace:
        t = e.get("turn")
        op = e.get("op", {}) or {}
        kind = op.get("op", "?")
        flow = list(turn_block(kind, t))
        flow.append(Paragraph("ACTION", lbl))
        flow.append(Paragraph(esc(op_summary(op)), action_st))
        flow.append(Paragraph(esc(json.dumps(op, ensure_ascii=False)), mono))
        fb = e.get("feedback", "")
        if fb:
            flow.append(Paragraph("RESULT", lbl))
            flow.append(Paragraph(esc(fb), result_st))
        rz = _raw(run_dir, t)
        if rz:
            flow.append(Paragraph("REASONING", lbl))
            flow.append(Paragraph(esc(rz).replace("\n", "<br/>"), reason_st))
        # memory (K) after this turn
        flow.append(Paragraph("MEMORY K — AFTER THIS TURN", lbl))
        header, rows = k_grid(k_after(run_dir, t))
        if header:
            tbl = Table([header] + rows, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "Courier", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 1), (0, -1), colors.HexColor(MUT)),
                ("TEXTCOLOR", (1, 1), (-1, -1), colors.HexColor(GREEN)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            flow.append(tbl)
        else:
            flow.append(Paragraph("<font color='#9ca3af'>K is empty</font>", small))
        # screenshots
        vtop = os.path.join(run_dir, f"turn_{t}_vtop.png")
        if os.path.exists(vtop):
            flow.append(Paragraph("TRUSTED V⊤ — what the actuator saw", cap))
            flow.append(_pdf_image(vtop, CONTENT_W))
        if kind == "query":
            outc = dict(re.findall(r"#(\d+):(\w+)", fb))
            vgs = sorted(glob.glob(os.path.join(run_dir, f"turn_{t}_mark*_vg.png")),
                         key=lambda p: int(re.search(r"mark(\d+)_", p).group(1)))
            cells, caps = [], []
            for p in vgs:
                m = re.search(r"mark(\d+)_", p).group(1)
                cells.append(_pdf_image(p, (CONTENT_W - 24) / 4))
                caps.append(Paragraph(f"#{m} &rarr; {esc(outc.get(m,'?'))}", cap))
            for i in range(0, len(cells), 4):
                flow.append(Paragraph("UNTRUSTED V_g — one mark revealed per image", cap))
                grid = Table([cells[i:i + 4], caps[i:i + 4]], hAlign="LEFT")
                grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                          ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
                flow.append(grid)
        after = os.path.join(run_dir, f"turn_{t}_after.png")
        if kind == "act" and os.path.exists(after):
            flow.append(Paragraph("result after act", cap))
            flow.append(_pdf_image(after, CONTENT_W))
        story.extend(flow)
        story.append(Spacer(1, 8))

    # baseline (no op_trace): render the plain CUA trajectory — reasoning+action + the screen the agent saw
    if not trace:
        traj = {}
        tp = os.path.join(run_dir, "trajectory.json")
        if os.path.exists(tp):
            try:
                traj = json.load(open(tp, encoding="utf-8"))
            except Exception:
                traj = {}
        for s in traj.get("steps", []) or []:
            i = s.get("step")
            flow = [Spacer(1, 8), Paragraph(f"<font color='#111827'><b>Step {i}</b></font>", th_st),
                    HRFlowable(width="100%", thickness=1.1, color=colors.HexColor("#1d4ed8"),
                               spaceBefore=1, spaceAfter=3)]
            resp = (s.get("model_response") or "").strip()
            if resp:
                flow.append(Paragraph("REASONING + ACTION", lbl))
                flow.append(Paragraph(esc(resp).replace("\n", "<br/>"), reason_st))
            acts = s.get("parsed_actions") or []
            if acts:
                flow.append(Paragraph("PARSED ACTION", lbl))
                flow.append(Paragraph(esc("; ".join(acts)), mono))
            obs = os.path.join(run_dir, f"step_{i - 1}.png")
            if os.path.exists(obs):
                flow.append(Paragraph("SCREEN THE AGENT SAW", cap))
                flow.append(_pdf_image(obs, CONTENT_W))
            story.extend(flow)
            story.append(Spacer(1, 8))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title=f"{task} secure run")
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else d.rstrip("/") + ".pdf"
    open(out, "wb").write(build_run_pdf(d))
    print("wrote", out)
