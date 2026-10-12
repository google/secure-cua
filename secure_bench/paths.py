
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

"""Central path resolution — the ONE module that knows where things live.

The bundled CODE (this package, the masking extension, the vendored CaMeL) and the results dir are
RELATIVE paths auto-derived from this file's own location — clone the repo anywhere and they resolve, nothing
to configure. The generated/fetched DATA (WebArena harness clone, Playwright browsers, agisdk scorer venv,
auth states) defaults to `fetched/` inside the repo but is relocatable off a monitored/quota'd code drive via
the SGE_FETCHED env var. A few other machine-specific things read an optional env var with a sensible default:
the Docker socket (DOCKER_HOST), the Vertex ADC credential (GOOGLE_APPLICATION_CREDENTIALS), and the Playwright
browser cache (PLAYWRIGHT_BROWSERS_PATH). Standard-library only, so this imports cleanly from every venv (the
secure_bench runners, the webarena_official eval subprocess, and the .agisdk_venv scorer).
"""
import os
import sys

# Artifact root = parent of secure_bench/ (this file is secure_bench/paths.py). Fully relative — no config.
SGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# --- everything below is INSIDE the artifact: pure relative, zero configuration ---
SECURE_BENCH      = os.path.join(SGE_ROOT, "secure_bench")
CAMEL_ROOT        = os.path.join(SGE_ROOT, "camel", "camel-prompt-injection-main")
EXT_DIR           = os.path.join(SGE_ROOT, "extension", "mask_extension")
RESULTS           = os.path.join(SGE_ROOT, "results")
ENV_FILE          = os.path.join(SGE_ROOT, ".env")
# Generated/installed artifacts (WebArena harness clone, agisdk venv, Playwright browsers, auth states).
# Default INSIDE the artifact (fetched/), but RELOCATABLE off a monitored/quota'd code drive: set SGE_FETCHED
# (e.g. in .env) to a writable local dir and everything below moves there. Only env var needed to relocate.
FETCHED           = os.environ.get("SGE_FETCHED") or os.path.join(SGE_ROOT, "fetched")
WEBARENA_OFFICIAL = os.path.join(FETCHED, "webarena_official")
# The official WebArena harness now runs in the PRIMARY conda env (merged 2026-07-27: the openai.error shim in
# wa_official_eval.py lets the unmodified harness import under a modern openai). The eval subprocess uses the
# SAME interpreter as the runner that spawns it — so this works with whichever conda env is active.
WEBARENA_VENV_PY  = sys.executable
AGISDK_VENV       = os.path.join(FETCHED, ".agisdk_venv")
AGISDK_VENV_PY    = os.path.join(AGISDK_VENV, "bin", "python")
WA_STATE_DIR      = os.path.join(FETCHED, "webarena")   # auth states (browser_env.auto_login)
WA_STATE = {
    "reddit":         os.path.join(WA_STATE_DIR, "reddit_state.json"),
    "shopping":       os.path.join(WA_STATE_DIR, ".auth", "shopping_state.json"),
    "shopping_admin": os.path.join(WA_STATE_DIR, "shopping_admin_state.json"),
    "gitlab":         os.path.join(WA_STATE_DIR, "gitlab_state.json"),
}

# --- genuinely machine-specific: optional env override, sensible default (still nothing to hand-edit) ---
def _docker_host():
    v = os.environ.get("DOCKER_HOST")
    if v:
        return v
    rootless = "/run/user/%d/docker.sock" % os.getuid()          # rootless docker (this repo's usual setup)
    return "unix://" + rootless if os.path.exists(rootless) else "unix:///var/run/docker.sock"  # else system daemon
DOCKER_HOST = _docker_host()
ADC_JSON    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(SGE_ROOT, "adc.json")
PLAYWRIGHT_BROWSERS_PATH = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") \
    or os.path.join(FETCHED, ".pw-browsers")
