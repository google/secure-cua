from pathlib import Path
from typing import Annotated

from pydantic import Field

from agentdojo.functions_runtime import TaskEnvironment, make_function, Depends
from agentdojo.task_suite.task_suite import TaskSuite

# omnizon/REAL + WebArena tasks build their UI via build_ui_webarena (Playwright/CDP).
from camel_cua.ui_benchmark.suites.base_ui_anthropic import (
    BaseUI_Anthropic,
    # Mouse actions
    left_single,
    left_double, 
    right_single,
    left_triple,
    middle_click,
    left_press,
    left_mouse_down,
    left_mouse_up,
    hover,
    drag,
    scroll,
    type_text,
    key,
    hold_key,
    hotkey,
    press,
    find,
    run_single_claude,
    check_done,
    summarize_screenshot_content,
    wait,
    mark_done,
    mark_fail,
    reset_memory,
    get_google_credentials,
    no_op,
    get_page_elements,
    get_page_text,
    find_element_by_text, 
    verify_hypothesis
)
def build_ui_anthropic(
        defense_level: int = 0,
        q_llm: str | None = None,
        q_llm_second_check: str | None = None,
        path_to_vm: str | None = None,
        path_to_vlm: str | None = "claude-4-5-sonnet-20250929",
        token_count_file: str | None = None,
    ) -> BaseUI_Anthropic:
    # Not implemented as a standalone builder — this stays only as the Depends() marker for
    # UIEnvironment_Anthropic below. omnizon/REAL + WebArena tasks build their UI via build_ui_webarena().
    raise RuntimeError(
        "build_ui_anthropic is not implemented — omnizon/REAL and WebArena tasks build their UI via "
        "build_ui_webarena(realbench=True).")


class UIEnvironment_Anthropic(TaskEnvironment):
    base_ui: Annotated[
        BaseUI_Anthropic,
        Depends(build_ui_anthropic),
    ] = Field(default=None, exclude=True)
    
    def model_copy(self, *, deep: bool = False) -> "UIEnvironment_Anthropic":
        # always do a shallow copy, even if deep=True
        return super().model_copy(deep=False)



TOOLS_0 = [
    # Mouse actions
    left_single,
    left_double,
    right_single,
    left_triple,
    middle_click,
    left_press,
    left_mouse_down,
    left_mouse_up,  
    hover,
    drag,
    scroll,
    type_text,
    key,
    hold_key,
    hotkey,
    press,
    find,
    run_single_claude,
    check_done,
    summarize_screenshot_content,
    wait,
    mark_done,
    mark_fail,
    reset_memory,
    get_google_credentials,
    no_op,
    get_page_elements,
    get_page_text,
    find_element_by_text,
    verify_hypothesis
]
DISALLOWED = {
    summarize_screenshot_content,
    get_page_elements,
    get_page_text,
    verify_hypothesis
}
TOOLS_1 = [tool for tool in TOOLS_0 if tool != run_single_claude]
TOOLS_2 = [tool for tool in TOOLS_1 if tool not in DISALLOWED]

TOOLS = {j: v for j, v in enumerate([TOOLS_0, TOOLS_1, TOOLS_2])}

HERE = Path(__file__).resolve().parent
SUITE_DATA = (
    HERE
    / ".."  # up from camel_cua/
    / ".."  # up from src/
    / "data"
    / "suites"
    / "realbench"
)
SUITE_DATA = SUITE_DATA.resolve()


def make_task_suite_Anthropic(
        system_defense_level: int = 0) -> TaskSuite[UIEnvironment_Anthropic]:
        
    return TaskSuite[UIEnvironment_Anthropic](
        "realbench",
        UIEnvironment_Anthropic,
        [make_function(tool) for tool in TOOLS[system_defense_level]],
        SUITE_DATA,
    )
