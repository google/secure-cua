#!/usr/bin/env bash

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

# Auto runtime environment for the artifact — sourced by every script in scripts/ and setup/.
# NOTHING here needs editing: SGE_ROOT is derived from this file's own location, .env supplies the
# credentials, and every other path is relative (see secure_bench/paths.py). Credentials are the only
# thing you provide (in .env).
SGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGE_ROOT

# credentials (the one file you fill in)
if [ -f "$SGE_ROOT/.env" ]; then set -a; . "$SGE_ROOT/.env"; set +a; fi

# Fetched DATA (WebArena harness clone, Playwright browsers, agisdk env, auth states). Defaults to fetched/
# inside the repo; set SGE_FETCHED (in .env) to a writable LOCAL dir to move it off a monitored/quota'd code
# drive. (The primary env is your activated conda env; see the block below.) Code stays in the repo.
export SGE_FETCHED="${SGE_FETCHED:-$SGE_ROOT/fetched}"

# ALL our code on the import path (no pip-install of the local packages needed — "no-build" install):
#   secure_bench (runners/core) · camel src (camel + camel_cua) · the WebArena harness
#   (browser_env/evaluation_harness/llms)
export PYTHONPATH="$SGE_ROOT/secure_bench:$SGE_ROOT/camel/camel-prompt-injection-main/src:$SGE_FETCHED/webarena_official${PYTHONPATH:+:$PYTHONPATH}"

# the primary interpreter (has playwright/anthropic/openai/genai + deps for the WebArena harness).
# NOT a config value — it's the python of your ACTIVATED conda env ($CONDA_PREFIX, from `conda activate sge`).
# (A pre-exported SGE_VENV is still honored, e.g. for headless/cron.)
if [ -z "${SGE_VENV:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    export SGE_VENV="$CONDA_PREFIX/bin/python"               # the activated (non-base) conda env
  else
    echo "env.sh: no conda env active — run 'conda activate sge' first (see README §2)" >&2
  fi
fi

# runtime defaults (relative/auto; override in .env only if you must)
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$SGE_FETCHED/.pw-browsers}"
if [ -z "${DOCKER_HOST:-}" ]; then                       # rootless docker socket if present, else the system daemon
  if [ -S "/run/user/$(id -u)/docker.sock" ]; then export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
  else export DOCKER_HOST="unix:///var/run/docker.sock"; fi
fi
