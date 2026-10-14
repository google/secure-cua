"""WebArena task-suite factory for CaMeL-CUA — parallels base_ui_task_suite_anthropic.py; the env is a
Playwright/CDP WebArena browser (BaseUI_WebArena). Reuses the anthropic TOOLS tiers +
SUITE_DATA verbatim (tools are duck-typed via Depends("base_ui"), so they bind to BaseUI_WebArena directly).
"""
import os
import sys
from typing import Annotated

from pydantic import Field

from anthropic import Anthropic, AnthropicVertex

from agentdojo.functions_runtime import TaskEnvironment, make_function, Depends
from agentdojo.task_suite.task_suite import TaskSuite

from camel_cua.vendored_prompts import SYSTEM_PROMPT

from camel_cua.ui_benchmark.suites.base_ui_webarena import BaseUI_WebArena, WebArenaEnvShim, _PWThread
from camel_cua.agent_pipeline_vlms.anthropic_vlm import ClaudeVLM
# reuse the anthropic tool tiers + suite data (identical tool set; tools are env-duck-typed)
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_anthropic import TOOLS, SUITE_DATA

REAL_BRIDGE = os.environ.get("SGE_ROOT") and os.path.join(os.environ["SGE_ROOT"], "secure_bench")
if not (REAL_BRIDGE and os.path.isdir(REAL_BRIDGE)):                       # else walk up to the artifact root
    _d = os.path.dirname(os.path.abspath(__file__))
    while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "secure_bench")):
        _d = os.path.dirname(_d)
    REAL_BRIDGE = os.path.join(_d, "secure_bench")
if REAL_BRIDGE not in sys.path:
    sys.path.insert(0, REAL_BRIDGE)
from wa_camel_helpers import CHROMIUM_EXE, _login_cookies, _site, VIEWPORT  # noqa: E402


def build_ui_webarena(
        defense_level: int = 0,
        q_llm: str | None = None,
        q_llm_second_check: str | None = None,
        path_to_vm: str | None = None,
        path_to_vlm: str | None = "claude-sonnet-4-5@20250929",
        token_count_file: str | None = None,
        *,
        task_spec: dict | None = None,
        out_dir: str | None = None,
        realbench: bool = False,        # realbench (omnizon): anonymous public site + a CDP debug port for agisdk
) -> BaseUI_WebArena:
    # task_spec/out_dir are required and supplied by WebArenaUIUserTask.init_environment. (The agentdojo field
    # `Depends(build_ui_webarena)` is vestigial — base_ui is set by init_environment, so this is never called
    # by the DI machinery; the guard just makes a misuse obvious instead of building a browser with no task.)
    if task_spec is None or out_dir is None:
        raise RuntimeError("build_ui_webarena requires task_spec + out_dir (set by WebArenaUIUserTask.init_environment)")

    # --- action VLM: Sonnet native computer-use grounder (identical to build_ui_anthropic:73-96) ---
    _vlm_model = path_to_vlm
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes"):
        client = AnthropicVertex(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            region=os.getenv("ANTHROPIC_VERTEX_REGION") or os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            timeout=300.0,
        )
        _parts = (path_to_vlm or "").rsplit("-", 1)
        if len(_parts) == 2 and _parts[1].isdigit() and len(_parts[1]) == 8:
            _vlm_model = _parts[0] + "@" + _parts[1]
    else:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=300.0)
    vlm = ClaudeVLM(
        client=client, model=_vlm_model, enable_tools=False, system=SYSTEM_PROMPT,
        display_width_px=1280, display_height_px=720,
    )

    # --- launch the WebArena browser in a DEDICATED Playwright thread (so its asyncio loop stays OFF the main
    #     thread, where CaMeL's Q-LLM async calls run). Launch pattern mirrors run_baseline_webarena.py:299-313. ---
    task = task_spec["webarena_task"]
    start_url = task_spec["start_url"]
    site = task_spec.get("site") or _site(task)
    profile_dir = os.path.join(out_dir, "_chrome_profile")
    os.makedirs(profile_dir, exist_ok=True)
    cookies = [] if realbench else _login_cookies(task)             # realbench omnizon is an anonymous public site
    debug_port = None
    if realbench:
        import socket
        _s = socket.socket(); _s.bind(("127.0.0.1", 0)); debug_port = _s.getsockname()[1]; _s.close()

    def _launch(pw):
        extra = [f"--remote-debugging-port={debug_port}"] if realbench else []   # agisdk attaches over this
        ctx = pw.chromium.launch_persistent_context(
            profile_dir, headless=False, executable_path=CHROMIUM_EXE,
            args=["--no-sandbox", "--disable-dev-shm-usage", *extra],
            viewport={"width": VIEWPORT[0], "height": VIEWPORT[1]}, device_scale_factor=1)
        if cookies:
            ctx.add_cookies(cookies)                                 # login BEFORE first nav
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        cdp = ctx.new_cdp_session(pg)
        pg.goto(start_url, wait_until="load")
        return ctx, pg, cdp

    pwt = _PWThread(_launch)
    pwt.start()
    pwt.wait_ready()
    shim = WebArenaEnvShim(pwt, start_url=start_url, site=site)
    if realbench:
        shim.cdp_http = f"http://127.0.0.1:{debug_port}"            # agisdk config/finisheval CDP target
    ui = BaseUI_WebArena(env=shim, vlm=vlm)
    ui.model_defense_level = defense_level
    ui.q_llm = q_llm
    ui.q_llm_second_check = q_llm_second_check
    ui.defense_map = {
        0: ui.model_defense_0, 1: ui.model_defense_1, 2: ui.model_defense_2,
        3: ui.model_defense_3, 4: ui.model_defense_4,
    }
    ui.token_count_file = token_count_file
    ui.path = out_dir
    ui.start_url = start_url
    ui.webarena_task = task
    ui.site = site
    return ui


class UIEnvironment_WebArena(TaskEnvironment):
    base_ui: Annotated[
        BaseUI_WebArena,
        Depends(build_ui_webarena),
    ] = Field(default=None, exclude=True)

    def model_copy(self, *, deep: bool = False) -> "UIEnvironment_WebArena":
        # the Playwright page/cdp/context are not deep-copyable -> force shallow
        return super().model_copy(deep=False)


def make_task_suite_WebArena(system_defense_level: int = 0) -> TaskSuite[UIEnvironment_WebArena]:
    # suite name "realbench" so make_tools_pipeline's realbench policy/prompt branch (models.py:203/205) fires;
    # TOOLS/SUITE_DATA reused from the anthropic suite (same tool set, same trivial environment.yaml).
    return TaskSuite[UIEnvironment_WebArena](
        "realbench",
        UIEnvironment_WebArena,
        [make_function(tool) for tool in TOOLS[system_defense_level]],
        SUITE_DATA,
    )
