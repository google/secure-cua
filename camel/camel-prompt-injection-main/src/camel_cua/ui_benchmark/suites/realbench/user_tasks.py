from pathlib import Path

from agentdojo.base_tasks import BaseUserTask
from agentdojo.functions_runtime import FunctionCall

from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_anthropic import UIEnvironment_Anthropic
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_webarena import UIEnvironment_WebArena

UIEnv = UIEnvironment_Anthropic | UIEnvironment_WebArena

# Re-exported by camel_cua.ui_benchmark.suites.realbench.__init__. The per-task registrars
# (register_omnizon_tasks / register_webarena_tasks) build their own CLI→suite-id maps in main.py,
# so this module-level mapping stays empty.
CLI_TO_SUITE_ID: dict[str, str] = {}


class UIUserTask(BaseUserTask[UIEnv]):
    """Base class for realbench/webarena UI tasks.

    Subclasses (OmnizonUIUserTask, WebArenaUIUserTask) skip this ``__init__`` and override
    ``init_environment``/``utility`` to build their in-memory example and Playwright backend;
    this base only carries the shared per-task attributes (assigned in main.py) and a default
    empty ``ground_truth``.
    """
    task_file: Path
    model_defense_level: int = 3  # default; can be overridden per task
    q_llm: str | None = None
    q_llm_second_check: str | None = None
    token_count_file: str | None = None
    path_to_vlm: str | None = None
    path_to_vm: str | None = None
    port: int = 15001
    plan_analysis_only: bool = False

    def ground_truth(self, pre_environment: UIEnv) -> list[FunctionCall]:
        return []
