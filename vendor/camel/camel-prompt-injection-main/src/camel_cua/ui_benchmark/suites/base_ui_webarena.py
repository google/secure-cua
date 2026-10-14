"""WebArena (Playwright/CDP) backend for the CaMeL-CUA agent — the analogue of base_ui_anthropic's
Playwright/CDP screenshot+a11y env, driving the WebArena Playwright container stack.

CaMeL's every GUI op funnels through 3 choke points in BaseUI_Anthropic: screenshot(), accessibility_tree()
and _step_with_retry()->env.step(pyautogui_str). Here we override screenshot()/accessibility_tree() and route
env.step()/teardown through WebArenaEnvShim.

CRITICAL — Playwright's SYNC api keeps an asyncio loop "running" in whatever thread it lives in, which
collides with CaMeL's Q-LLM async calls (find_element_by_text/query_ai_assistant/verify_hypothesis do
loop.run_until_complete -> "event loop is already running"). So ALL Playwright work runs in a DEDICATED
thread (_PWThread); the main thread (interpreter + Q-LLM) stays asyncio-clean. screenshot/a11y/step/url/close
marshal callables to that thread.

Viewport 1920x1080 dsf=1 => getBoundingClientRect px == page.screenshot px == page.mouse px == CaMeL's OS-pixel
frame (one clean frame; find() grounding reused unchanged). Reuses only the stdlib leaves wa_workers + wa_a11y.
"""
import os
import io
import queue
import sys
import threading
import time as _time

import PIL.Image

from pydantic import Field

from camel_cua.ui_benchmark.suites.base_ui_anthropic import BaseUI_Anthropic, Screenshot, logger

REAL_BRIDGE = os.environ.get("SGE_ROOT") and os.path.join(os.environ["SGE_ROOT"], "secure_bench")
if not (REAL_BRIDGE and os.path.isdir(REAL_BRIDGE)):                       # else walk up to the artifact root
    _d = os.path.dirname(os.path.abspath(__file__))
    while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "secure_bench")):
        _d = os.path.dirname(_d)
    REAL_BRIDGE = os.path.join(_d, "secure_bench")
if REAL_BRIDGE not in sys.path:
    sys.path.insert(0, REAL_BRIDGE)
from wa_a11y import INTERACTIVE as _INTERACTIVE, _SKIP, _rect          # noqa: E402


# --- pyautogui-style key/combo -> Playwright key (replicated from run_secure_webarena._pwkey) --------
_KEYMAP = {"enter": "Enter", "return": "Enter", "tab": "Tab", "backspace": "Backspace", "esc": "Escape",
           "escape": "Escape", "space": "Space", " ": "Space", "delete": "Delete", "del": "Delete",
           "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown", "up": "ArrowUp",
           "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
           "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight"}
_MODS = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "command": "Meta", "meta": "Meta",
         "alt": "Alt", "option": "Alt", "shift": "Shift"}


def _pwkey(k):
    k = str(k).strip()
    if "+" in k:
        out = []
        for p in [p.strip() for p in k.split("+") if p.strip()]:
            pl = p.lower()
            out.append(_MODS.get(pl, _KEYMAP.get(pl, p if len(p) == 1 else p.capitalize())))
        return "+".join(out)
    kl = k.lower()
    return _KEYMAP.get(kl, k if len(k) == 1 else k.capitalize())


# --- CDP AX role -> AT-SPI tag vocabulary CaMeL's a11y filters match --------------------------------
_ROLE2TAG = {
    "button": "push-button", "link": "link", "textbox": "entry", "searchbox": "entry",
    "checkbox": "checkbox", "radio": "checkbox", "menuitemcheckbox": "checkbox", "menuitemradio": "checkbox",
    "switch": "toggle-button", "menuitem": "menuitem", "option": "menuitem",
    "combobox": "menu", "listbox": "menu", "menu": "menu", "tab": "push-button",
    "heading": "heading", "paragraph": "text", "StaticText": "text", "staticText": "text", "text": "text",
}


def _axtree_to_tsv(cdp, viewport=(1920, 1080), max_chars=200000, cap_rects=250):
    """Build the 7-col TSV (tag,name,text,class,description,position,size) CaMeL's linearizer produces, from CDP
    Accessibility.getFullAXTree. Boxes (getBoundingClientRect, viewport px) only for interactive roles (capped).
    ALWAYS 7 tab fields (find_element_by_text's len(parts)>=7 gate). MUST be called inside the Playwright thread."""
    cdp.send("DOM.enable")
    cdp.send("Accessibility.enable")
    tree = cdp.send("Accessibility.getFullAXTree")
    VW, VH = viewport
    header = "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    lines = [header]
    nrect = 0
    for nd in tree.get("nodes", []):
        if nd.get("ignored"):
            continue
        role = (nd.get("role") or {}).get("value") or ""
        name = ((nd.get("name") or {}).get("value") or "").strip()
        interactive = role in _INTERACTIVE
        if role in _SKIP and not interactive:
            continue
        if not name and not interactive:
            continue
        tag = _ROLE2TAG.get(role, role)
        desc = ((nd.get("description") or {}).get("value") or "").strip()
        pos, size = "", ""
        bid = nd.get("backendDOMNodeId")
        if interactive and bid and nrect < cap_rects:
            nrect += 1
            box = _rect(cdp, bid)
            if box and (0 <= box[0] < VW and -40 < box[1] < VH + 40):
                pos, size = f"({box[0]}, {box[1]})", f"({box[2]}, {box[3]})"
        nm = name[:120].replace("\t", " ").replace("\n", " ")
        ds = desc[:120].replace("\t", " ").replace("\n", " ")
        lines.append(f'{tag}\t{nm}\t""\t\t{ds}\t{pos}\t{size}')
        if sum(len(x) + 1 for x in lines) > max_chars:
            break
    return "\n".join(lines)[:max_chars]


# --- pyautogui -> Playwright shim (executes CaMeL's own generated pyautogui strings) -----------------
class PyAutoGUIShim:
    """CaMeL's actuator methods (base_ui_anthropic.py:1947-2136) emit pyautogui strings; this maps that exact
    surface onto a sync Playwright page. Runs inside the Playwright thread. 1920x1080 viewport px."""
    _SCROLL_STEP = 120

    def __init__(self, page):
        self.page = page
        self._x, self._y = 960, 540

    def _mv(self, x=None, y=None):
        if x is not None and y is not None:
            self._x = max(0, min(1919, int(x)))
            self._y = max(0, min(1079, int(y)))
            self.page.mouse.move(self._x, self._y)

    @staticmethod
    def _btn(button):
        b = (button or "left").lower()
        return b if b in ("left", "right", "middle") else "left"

    def click(self, x=None, y=None, button="left", clicks=1, interval=0.0):
        self._mv(x, y)
        self.page.mouse.click(self._x, self._y, button=self._btn(button), click_count=int(clicks))

    def doubleClick(self, x=None, y=None, button="left"):
        self._mv(x, y)
        self.page.mouse.dblclick(self._x, self._y, button=self._btn(button))

    def middleClick(self, x=None, y=None, button="left"):
        self._mv(x, y)
        self.page.mouse.click(self._x, self._y, button="middle")

    def moveTo(self, x, y, duration=0.0):
        self._mv(x, y)

    def dragTo(self, x, y, duration=0.0, button="left"):
        self.page.mouse.down(button=self._btn(button))
        self._mv(x, y)
        self.page.mouse.up(button=self._btn(button))

    def mouseDown(self, x=None, y=None, button="left"):
        self._mv(x, y)
        self.page.mouse.down(button=self._btn(button))

    def mouseUp(self, x=None, y=None, button="left"):
        self._mv(x, y)
        self.page.mouse.up(button=self._btn(button))

    def scroll(self, amount, x=None, y=None):
        self._mv(x, y)
        self.page.mouse.wheel(0, -int(amount) * self._SCROLL_STEP)

    def hscroll(self, amount, x=None, y=None):
        self._mv(x, y)
        self.page.mouse.wheel(int(amount) * self._SCROLL_STEP, 0)

    def write(self, text, interval=0.0):
        self.page.keyboard.type(str(text), delay=int(float(interval) * 1000))

    def press(self, key):
        self.page.keyboard.press(_pwkey(key))

    def hotkey(self, *keys):
        self.page.keyboard.press("+".join(_pwkey(k) for k in keys))

    def keyDown(self, key):
        self.page.keyboard.down(_pwkey(key))

    def keyUp(self, key):
        self.page.keyboard.up(_pwkey(key))


def _run_pyautogui(code, page):
    """Exec CaMeL's wrapped pyautogui string (`import pyautogui\\nimport time\\n...`) against the Playwright
    shim — neutralize the literal imports (they'd import the real host-controlling pyautogui)."""
    if code.strip() == "WAIT":
        _time.sleep(1)
        return
    body = code.replace("import pyautogui", "pass").replace("import time", "pass")
    exec(body, {"pyautogui": PyAutoGUIShim(page), "time": _time})     # noqa: S102 — CaMeL-generated, not untrusted


# --- dedicated Playwright thread (owns pw/context/page/cdp; keeps its asyncio loop off the main thread) ---
class _PWThread(threading.Thread):
    def __init__(self, launch_fn):
        super().__init__(daemon=True)
        self._launch_fn = launch_fn                                  # (pw) -> (context, page, cdp), runs in-thread
        self._q = queue.Queue()
        self._ready = threading.Event()
        self.err = None
        self.pw = self.context = self.page = self.cdp = None

    def run(self):
        try:
            from playwright.sync_api import sync_playwright
            self.pw = sync_playwright().start()
            self.context, self.page, self.cdp = self._launch_fn(self.pw)
        except Exception as e:
            self.err = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._q.get()
            if item is None:                                         # teardown sentinel — close IN this thread
                try:
                    self.context.close()
                except Exception:
                    pass
                try:
                    self.pw.stop()
                except Exception:
                    pass
                return
            fn, fut = item
            try:
                fut["result"], fut["ok"] = fn(self.page, self.cdp), True
            except Exception as e:
                fut["error"], fut["ok"] = e, False
            fut["evt"].set()

    def wait_ready(self, timeout=90):
        if not self._ready.wait(timeout):
            raise RuntimeError("Playwright thread did not become ready")
        if self.err is not None:
            raise self.err

    def submit(self, fn):
        fut = {"evt": threading.Event(), "ok": None}
        self._q.put((fn, fut))
        fut["evt"].wait()
        if not fut["ok"]:
            raise fut["error"]
        return fut["result"]

    def shutdown(self):
        self._q.put(None)
        self.join(timeout=30)


class WebArenaEnvShim:
    """The `.env` handle BaseUI_WebArena holds. All Playwright work is marshalled to the _PWThread."""

    def __init__(self, pwt, start_url=None, site=None):
        self.pwt = pwt
        self.start_url = start_url
        self.site = site
        self.cdp_http = None          # realbench: the agisdk CDP endpoint (set by build_ui_webarena when realbench=True)

    # --- the three env operations CaMeL touches ---
    def screenshot_bytes(self):
        def _shot(pg, cdp):
            # Wait for the compositor to commit a PAINTED frame before capturing. `pg.goto(wait_until="load")`
            # returns on the load event, which can fire before first paint, so a capture taken right after a
            # navigation races the frame and Chromium's CDP Page.captureScreenshot returns "Unable to capture
            # screenshot". A double requestAnimationFrame resolves only after the main thread has submitted a
            # frame, so by the time we capture there is always a frame to grab — the error can't occur.
            try:
                pg.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            except Exception:
                pass                                           # no JS context yet (rare) -> the guard below covers it
            try:
                return pg.screenshot()
            except Exception:                                  # last-resort net (should never trigger after the frame wait)
                try:
                    pg.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                return pg.screenshot()
        return self.pwt.submit(_shot)

    def a11y(self):
        return self.pwt.submit(lambda pg, cdp: _axtree_to_tsv(cdp))

    def a11y_ready(self, floor=500):
        """Content-agnostic readiness gate: the CDP-AXTree TSV exceeds `floor` chars."""
        try:
            return len(self.a11y()) >= floor
        except Exception:
            return False

    def step(self, code, pause=0):
        self.pwt.submit(lambda pg, cdp: _run_pyautogui(code, pg))
        return None, None, False, {}

    def url(self):
        try:
            return self.pwt.submit(lambda pg, cdp: pg.url)
        except Exception:
            return ""

    def evaluate(self):
        return 0.0                                                   # done-check never early-stops; real eval external

    def close(self):
        try:
            self.pwt.shutdown()
        except Exception:
            pass

    def reset(self, task_config=None):
        if self.start_url:
            self.pwt.submit(lambda pg, cdp: pg.goto(self.start_url, wait_until="load"))


class BaseUI_WebArena(BaseUI_Anthropic):
    """CaMeL BaseUI backed by Playwright/CDP. Overrides only screenshot()/accessibility_tree(); _step_with_retry
    inherited (its env.step lands on WebArenaEnvShim.step). find(), all actuators, find_element_by_text, defenses
    inherited unchanged (tools duck-typed)."""

    env: WebArenaEnvShim = Field(exclude=True, repr=False)
    start_url: str | None = Field(default=None, exclude=True)
    webarena_task: str | None = Field(default=None, exclude=True)
    site: str | None = Field(default=None, exclude=True)

    def screenshot(self, add: bool = False) -> Screenshot:
        raw = self.env.screenshot_bytes()
        img = PIL.Image.open(io.BytesIO(raw))
        if add:
            self.history_images.append(img)
            if len(self.history_images) > self.max_history:
                self.history_images.pop(0)
        self.image_height, self.image_width = img.height, img.width
        self.resize_factor = (self.image_width / float(self.tool_display_w),
                              self.image_height / float(self.tool_display_h))
        return Screenshot.from_pil(img)

    def accessibility_tree(self, add: bool = False) -> str:
        try:
            tree = self.env.a11y()
        except Exception as e:
            logger.warning(f"[accessibility_tree/webarena] failed: {e!r}")
            return "[DOM unavailable - fetch failed]"
        if add:
            self.history_dom.append(tree)
            if len(self.history_dom) > self.max_history:
                self.history_dom.pop(0)
        return tree
