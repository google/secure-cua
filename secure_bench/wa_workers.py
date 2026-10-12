
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

"""Same-site parallelism helpers (integrity-safe, single source of truth).

To run K tasks on ONE site concurrently, each task's WORKER runs its OWN container on a derived port, so it
keeps per-task reset + mutation isolation (you cannot share one container across parallel tasks that reset it).
A worker is identified by two env vars set by the parallel launcher:
    WA_SITE       = the site this worker handles (reddit|shopping|shopping_admin|gitlab)
    WA_SITE_PORT  = the external port of this worker's container (base + 10000*i)
Everything a worker touches — start/eval URL, container name, auth-state file, mask seed host — derives from
those. When they are UNSET (or WA_SITE != the queried site) every helper returns the site's CANONICAL value,
so behaviour is BYTE-IDENTICAL to the pre-parallelism single-container harness (the running arm is unaffected).

Port formula `base + 10000*i` MUST stay in sync with mask_extension/build_seed.js (which seeds mask rules
under the same worker ports — masking silently fails on an unseeded port).
"""
import os

SITE_DEFAULT_PORT = {"reddit": 9999, "shopping": 7770, "shopping_admin": 7780, "gitlab": 8023}
MAX_WORKERS = 6                                    # base + 5 workers; keep <= build_seed.js MAX_WORKERS


def site_of(task):                                 # "shopping_admin-208" -> "shopping_admin"
    return task.rsplit("-", 1)[0]


def worker_port(site):
    """Port THIS process should target for `site`. WA_SITE_PORT overrides ONLY for the worker's own WA_SITE
    (legacy: a bare WA_SITE_PORT with no WA_SITE meant reddit — preserved). Otherwise the canonical port."""
    wport = os.getenv("WA_SITE_PORT")
    if not wport:
        return SITE_DEFAULT_PORT[site]
    wsite = os.getenv("WA_SITE", "reddit")         # legacy bare WA_SITE_PORT == reddit
    return int(wport) if wsite == site else SITE_DEFAULT_PORT[site]


def is_worker(site):
    return worker_port(site) != SITE_DEFAULT_PORT[site]


def worker_ports(site, k):
    """The k container ports for a site's workers: [base, base+10000, ... ] (k<=MAX_WORKERS)."""
    base = SITE_DEFAULT_PORT[site]
    return [base + 10000 * i for i in range(min(k, MAX_WORKERS))]


def rewrite_port(url, site):
    """Rewrite the site's canonical port -> this worker's port in a URL (identity for the base container)."""
    d, p = SITE_DEFAULT_PORT[site], worker_port(site)
    return url if p == d else url.replace(f"localhost:{d}", f"localhost:{p}")


def worker_state_path(site, base_state):
    """Per-worker auth file. Session-based sites (shopping/gitlab/shopping_admin) validate the cookie against
    the CONTAINER's session store, which is fresh per worker container -> each worker needs its own file
    (a shared file would race as workers relogin). Base container -> the unchanged base file."""
    if not is_worker(site):
        return base_state
    root, ext = os.path.splitext(base_state)
    return f"{root}.p{worker_port(site)}{ext}"


def worker_container(site, base_name):
    """Per-worker container name (base container keeps its name)."""
    return base_name if not is_worker(site) else f"{base_name}-p{worker_port(site)}"
