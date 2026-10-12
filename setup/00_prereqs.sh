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

# Step 0 — ensure host prerequisites are INSTALLED (assumes root or sudo). Installs anything missing via the
# system package manager. Set CHECK_ONLY=1 to only report without installing.
set -uo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
CHECK_ONLY="${CHECK_ONLY:-0}"

if   command -v apt-get >/dev/null 2>&1; then PM=apt
elif command -v dnf     >/dev/null 2>&1; then PM=dnf
elif command -v yum     >/dev/null 2>&1; then PM=yum
elif command -v pacman  >/dev/null 2>&1; then PM=pacman
else PM=""; fi
echo "### prerequisites (package manager: ${PM:-none}, sudo: ${SUDO:-none}) ###"

need () { command -v "$1" >/dev/null 2>&1; }
pkg () {  # pkg <apt-name> <dnf/yum-name> <pacman-name>
  [ "$CHECK_ONLY" = 1 ] && { echo "    (check-only) would install: $1"; return 1; }
  case "$PM" in
    apt)    $SUDO apt-get update -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y "$1" ;;
    dnf)    $SUDO dnf install -y "$2" ;;
    yum)    $SUDO yum install -y "$2" ;;
    pacman) $SUDO pacman -Sy --noconfirm "$3" ;;
    *) echo "    no known package manager — install '$1' manually"; return 1 ;;
  esac
}
report () { need "$1" && echo "  [ok]  $2" || echo "  [!!]  $2  — install failed, do it manually"; }

# The primary env is a CONDA env (Python 3.10-3.12 comes from conda) — so we need conda, not a system python/venv.
if [ -n "${CONDA_PREFIX:-}" ] || need conda; then
  echo "  [ok]  conda${CONDA_DEFAULT_ENV:+ (active env: $CONDA_DEFAULT_ENV)}"
else
  echo "  [!!]  conda not found — install Miniconda (https://docs.conda.io/en/latest/miniconda.html), then:"
  echo "        conda create -y -n sge python=3.12 && conda activate sge   (see README §2)"
fi

need git   || { echo "installing git ...";   pkg git git git; }
report git "git"

need node  || { echo "installing Node.js ..."; pkg nodejs nodejs nodejs; }
report node "node ($(node -v 2>/dev/null))"

need Xvfb  || { echo "installing Xvfb ...";   pkg xvfb xorg-x11-server-Xvfb xorg-server-xvfb; }
report Xvfb "Xvfb"

if ! need docker; then
  echo "installing Docker (get.docker.com) ..."
  [ "$CHECK_ONLY" = 1 ] || { curl -fsSL https://get.docker.com | $SUDO sh; $SUDO systemctl enable --now docker 2>/dev/null || true; }
fi
if need docker && docker info >/dev/null 2>&1; then echo "  [ok]  docker (daemon reachable)"
elif need docker; then echo "  [!!]  docker installed but daemon not reachable — start it (systemctl start docker) / check DOCKER_HOST"
else echo "  [!!]  docker — install failed"; fi

# realbench (REAL/omnizon) needs no KVM/VM — it runs through the same Playwright pipeline as WebArena; its only
# extra is the agisdk scorer venv, built by setup/10 with SETUP_REALBENCH=1.

echo "### prerequisites done ###"
