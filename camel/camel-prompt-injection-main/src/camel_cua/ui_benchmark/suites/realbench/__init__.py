# Keep injection tasks if you use them
from .injection_tasks import *  # noqa: F401

# Public suites — only the two that are ever dispatched: Anthropic (claude) + WebArena/realbench (Playwright).
from .base_ui_task_suite_anthropic import make_task_suite_Anthropic
from .base_ui_task_suite_webarena import make_task_suite_WebArena
# Import the user_tasks module for its side-effects (auto-register)
from . import user_tasks as _user_tasks  # DO NOT star-import; just load module

# Re-export the CLI → suite-id map so main can import from the package
CLI_TO_SUITE_ID = getattr(_user_tasks, "CLI_TO_SUITE_ID", {})

__all__ = ["CLI_TO_SUITE_ID", "make_task_suite_Anthropic", "make_task_suite_WebArena"]
