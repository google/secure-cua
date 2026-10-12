
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

"""Stdlib-only WebArena helpers for the CaMeL-CUA WebArena backend (imported by the CaMeL venv).

A faithful copy of the leaf helpers in run_secure_webarena.py (CHROMIUM_EXE, _login_cookies, _official_task,
TASK_IDS/TASKS, _site) so the CaMeL backend can launch the same browser + load the same tasks + login WITHOUT
importing the heavy driver (which pulls in smask_runtime/secure_actuator/secure_shadow/secure_llm). Imports only
wa_workers (leaf) + stdlib. Kept in sync with run_secure_webarena.py — if the canonical helpers change, update here.
"""
import json
import os
import paths
import re

import wa_workers as ww

ROOT = paths.SGE_ROOT
WEBARENA_OFFICIAL = paths.WEBARENA_OFFICIAL
VIEWPORT = (1920, 1080)

REDDIT_STATE = paths.WA_STATE["reddit"]
SHOPPING_STATE = paths.WA_STATE["shopping"]
GITLAB_STATE = paths.WA_STATE["gitlab"]
SHOPPING_ADMIN_STATE = paths.WA_STATE["shopping_admin"]
SITE_STATE = {"reddit": REDDIT_STATE, "shopping": SHOPPING_STATE,
              "gitlab": GITLAB_STATE, "shopping_admin": SHOPPING_ADMIN_STATE}

# The 40-task WebArena set (10/site) — identical to run_secure_webarena.TASK_IDS so CaMeL runs the SAME tasks
# as the masking/baseline campaign (direct cross-defense comparison).
TASK_IDS = [
    "reddit-27", "reddit-66", "reddit-399", "reddit-404", "reddit-409",
    "reddit-580", "reddit-595", "reddit-605", "reddit-719", "reddit-725",
    "shopping-21", "shopping-163", "shopping-188", "shopping-269", "shopping-279",
    "shopping-324", "shopping-384", "shopping-465", "shopping-511", "shopping-585",
    "gitlab-102", "gitlab-132", "gitlab-168", "gitlab-308", "gitlab-314",
    "gitlab-339", "gitlab-389", "gitlab-411", "gitlab-441", "gitlab-784",
    "shopping_admin-0", "shopping_admin-112", "shopping_admin-198", "shopping_admin-208", "shopping_admin-243",
    "shopping_admin-458", "shopping_admin-464", "shopping_admin-470", "shopping_admin-694", "shopping_admin-704",
]


def _chromium_exe():
    """Newest installed Playwright chromium (chromium-1187 renderer-crashes on gitlab's Monaco; 1228 is fine)."""
    base = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "")
    best = (-1, None)
    if base and os.path.isdir(base):
        for d in os.listdir(base):
            m = re.fullmatch(r"chromium-(\d+)", d)
            if not m:
                continue
            for sub in ("chrome-linux64/chrome", "chrome-linux/chrome"):
                exe = os.path.join(base, d, sub)
                if os.path.exists(exe) and int(m.group(1)) > best[0]:
                    best = (int(m.group(1)), exe)
                    break
    return best[1]


CHROMIUM_EXE = _chromium_exe()


def _site(task):
    return task.rsplit("-", 1)[0]


def _login_cookies(task, log=print):
    """Login cookies for the task's SITE (worker-aware). Missing storage_state -> [] (anonymous)."""
    path = SITE_STATE.get(_site(task))
    if path:
        path = ww.worker_state_path(_site(task), path)
    if not path or not os.path.exists(path):
        log(f"[login] no storage_state for site '{_site(task)}' ({path}) -> ANONYMOUS")
        return []
    return json.load(open(path, encoding="utf-8")).get("cookies", [])


def _official_task(task):
    """goal + start_url loaded verbatim from the official resolved config; port rewritten to this worker."""
    cfg_path = os.path.join(WEBARENA_OFFICIAL, "config_files", f"{int(task.rsplit('-', 1)[1])}.json")
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"missing {cfg_path}")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    return {"goal": cfg["intent"], "start_url": ww.rewrite_port(cfg["start_url"], _site(task))}
