
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

"""Reset a WebArena container to a PRISTINE DB snapshot — MULTI-SITE (reddit, shopping; extensible).

Each webarena-verified image bakes the initial WebArena DB into its layers, so the reliable way to revert a
task's mutations between scored runs is to destroy the container and start a fresh one from the image (COW
filesystem => pristine baked DB). Per-site config in SITE_RESET. Readiness = site 200 + a content page renders
(+ services healthy where an env-ctrl panel exists) + the eval user is authenticated. NOTE: Magento sessions
do NOT survive a container swap (session store is fresh), so for `relogin` sites the storage_state is
regenerated (webarena auto_login) against the fresh container; reddit's REMEMBERME token persists so it isn't.

CLI:  python wa_reset.py --site reddit             # reset + report
      python wa_reset.py --site shopping --check    # readiness/login check only (no reset)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import paths
import shutil
import socket
import subprocess
import time
import urllib.request

os.environ.setdefault("DOCKER_HOST", paths.DOCKER_HOST)

ROOT = paths.SGE_ROOT
WEBARENA = paths.WEBARENA_OFFICIAL
_EVAL_PY = paths.WEBARENA_VENV_PY                                         # primary conda env (auto_login lives here)
# canonical site URLs the harness env_config asserts non-empty (needed by auto_login)
_URL_ENV = {"REDDIT": "http://localhost:9999", "SHOPPING": "http://localhost:7770",
            "SHOPPING_ADMIN": "http://localhost:7780/admin", "GITLAB": "http://localhost:8023",
            "WIKIPEDIA": "http://localhost:8888", "MAP": "http://localhost:3000",
            "HOMEPAGE": "http://localhost:4399"}

import wa_workers as ww                                                    # same-site parallelism: per-worker name/port/state
# For a worker suite the launcher sets WA_SITE + WA_SITE_PORT -> the worker's container name, external port and
# auth-state file all derive from them; with neither set (base run / the running arm) every value is the site's
# canonical one, so reset() behaves EXACTLY as before.

# Per-site reset config. content_path+marker prove the DB rendered; ctrl_port!=None -> also poll env-ctrl
# /status; login_markers prove the eval user is authed; relogin -> regenerate storage_state after the swap.
# OFFICIAL WebArena images (2026-07-24). Magento/gitlab base_url/external_url are baked into a *_configured
# snapshot (docker commit after standup) so a fresh container is pristine-DB AND correctly configured — no
# re-standup per reset. `internal_port` (default 80) + `run_extra` (entrypoint args) handle gitlab's shape.
# `login_path` is where check_ready probes the eval-user's auth (default "/").
SITE_RESET = {
    "reddit": {
        "name": ww.worker_container("reddit", "webarena-verified-reddit"),
        "image": "postmill-populated-exposed-withimg:latest",
        "site_port": str(ww.worker_port("reddit")), "internal_port": "80", "ctrl_port": None,
        "state": ww.worker_state_path("reddit", paths.WA_STATE["reddit"]),
        "content_path": "/f/books", "content_marker": "submission", "login_path": "/",
        "login_markers": ["MarvelsGrantMan136", "logout"], "relogin": True,
    },
    "shopping": {
        "name": ww.worker_container("shopping", "webarena-shopping"),
        "image": "shopping_configured:latest",
        "site_port": str(ww.worker_port("shopping")), "internal_port": "80", "ctrl_port": None,  # Magento: no env-ctrl /status
        "state": ww.worker_state_path("shopping", paths.WA_STATE["shopping"]),
        "content_path": "/", "content_marker": "catalogsearch", "login_path": "/",
        "login_markers": ["Sign Out", "Welcome, Emma"], "relogin": True,  # session store is fresh post-swap -> re-login
    },
    "shopping_admin": {
        "name": ww.worker_container("shopping_admin", "webarena-shopping-admin"),
        "image": "shopping_admin_configured:latest",
        "site_port": str(ww.worker_port("shopping_admin")), "internal_port": "80", "ctrl_port": None,
        "state": ww.worker_state_path("shopping_admin", paths.WA_STATE["shopping_admin"]),
        "content_path": "/admin", "content_marker": "login", "login_path": "/admin",
        "login_markers": ["Magento Admin", "menu-magento-backend-dashboard", "adminhtml"], "relogin": True,
    },
    "gitlab": {
        "name": ww.worker_container("gitlab", "webarena-verified-gitlab"),
        "image": "gitlab_configured:latest",
        "site_port": str(ww.worker_port("gitlab")), "internal_port": "8023", "ctrl_port": None,
        "run_extra": ["/opt/gitlab/embedded/bin/runsvdir-start"],        # gitlab entrypoint
        "state": ww.worker_state_path("gitlab", paths.WA_STATE["gitlab"]),
        "content_path": "/explore", "content_marker": "gitlab", "login_path": "/",
        "login_markers": ["byteblaze", "Dashboard"], "relogin": True, "timeout": 420,
    },
}


_Res = collections.namedtuple("_Res", "returncode stdout stderr")

# stderr substrings that mean "the daemon was momentarily busy/unreachable, or the port hasn't been released
# yet" — i.e. RETRY is worth it. A real error (bad image, syntax) is NOT in here, so it fails fast.
_TRANSIENT = (
    "cannot connect to the docker daemon", "is the docker daemon running", "error during connect",
    "connection refused", "i/o timeout", "deadline exceeded", "context canceled",
    "resource temporarily unavailable", "device or resource busy", "resource busy",
    "port is already allocated", "address already in use",
    "failed to set up container networking", "500 internal server error",
)
# NOTE: a NAME conflict ("already in use by container") is NOT here — nothing frees the name between inner
# retries, so reset()'s OUTER rm+run loop owns that remediation. Bare "eof"/"timeout" are excluded too: too
# short, they'd match unrelated real errors (e.g. mysql "lock wait timeout"). "i/o timeout"/"deadline exceeded"
# cover the genuine socket-hang cases.


def _docker(*args, timeout=120, retries=3):
    """Run a docker command, RETRYING transient daemon/socket/port failures with backoff. Never raises: a hung
    socket (TimeoutExpired) or any exception becomes a non-zero _Res so callers stay on the returncode/stderr
    contract. Non-transient failures return immediately (no wasted retries)."""
    last = _Res(125, "", "docker: not run")
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return _Res(0, r.stdout, r.stderr)
            last = _Res(r.returncode, r.stdout, r.stderr)
            if not any(s in (r.stderr or "").lower() for s in _TRANSIENT):
                return last                                          # real error -> fail fast
        except subprocess.TimeoutExpired:
            last = _Res(124, "", f"docker {' '.join(map(str, args[:2]))} timed out after {timeout}s (daemon busy?)")
        except Exception as e:                                       # socket gone, OS error, etc.
            last = _Res(125, "", str(e))
        if attempt < retries:
            time.sleep(min(8.0, 1.5 * (attempt + 1)))               # backoff: 1.5, 3, 4.5, ...
    return last


def _daemon_ok(timeout=20):
    """True if the docker daemon answers. Bounds the socket-hang case so we surface a clear error instead of
    blocking a whole task on a dead daemon."""
    return _docker("version", "--format", "{{.Server.Version}}", timeout=timeout, retries=2).returncode == 0


def _port_free(port):
    """True if `port` can be bound right now. SO_REUSEADDR so a lingering TIME_WAIT from a just-removed
    container (or a health probe) does NOT read as 'busy' — that false-positive is what made the swap slow."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_ports_free(ports, timeout=25, poll=1.0):
    """Wait until every port is bindable (the old container's port has actually been released by the daemon)."""
    t0 = time.time()
    ports = [p for p in ports if p]
    while time.time() - t0 < timeout:
        if all(_port_free(p) for p in ports):
            return True
        time.sleep(poll)
    return all(_port_free(p) for p in ports)


def _get(url, timeout=6, cookies=None):
    req = urllib.request.Request(url)
    if cookies:
        req.add_header("Cookie", cookies)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return None, str(e)


def _cookie_header(state_file):
    """Persistent cookies from the saved storage_state, as a Cookie header string."""
    try:
        cks = json.load(open(state_file, encoding="utf-8")).get("cookies", [])
        return "; ".join(f"{c['name']}={c['value']}" for c in cks)
    except Exception:
        return ""


def _services_healthy(ctrl_port):
    code, body = _get(f"http://localhost:{ctrl_port}/status")
    if code != 200:
        return False, {}
    try:
        svc = json.loads(body)["details"]["value"]["services"]
    except Exception:
        return False, {}
    ok = (svc.get("postgres") == "HEALTHY" and svc.get("nginx") == "RUNNING"
          and svc.get("php-fpm") == "RUNNING")
    return ok, svc


def check_ready(site="reddit", want_login=True):
    """Return (ready, detail). Ready = site 200 + services healthy (if ctrl) + content renders + (login re-auth)."""
    c = SITE_RESET[site]
    base = f"http://localhost:{c['site_port']}"
    site_code, _ = _get(base + "/")
    healthy, svc = _services_healthy(c["ctrl_port"]) if c["ctrl_port"] else (True, {})
    content, cbody = _get(base + c["content_path"])
    has_content = content == 200 and c["content_marker"].lower() in (cbody or "").lower()
    login_ok = True
    if want_login:
        _, lbody = _get(base + c.get("login_path", "/"), cookies=_cookie_header(c["state"]))
        login_ok = any(m.lower() in (lbody or "").lower() for m in c["login_markers"])
    ready = bool(site_code == 200 and healthy and has_content and login_ok)
    return ready, {"site": site_code, "services": svc, "has_content": has_content, "login_ok": login_ok}


def _regen_login(site, log=print):
    """Regenerate the site's storage_state on the FRESH container (its session didn't survive the swap).
    Worker-aware: auto_login targets THIS worker's container (its port) and stages the intermediate
    {site}_state.json in a PER-WORKER subfolder so parallel siblings never race on it, then copies to the
    worker's state file. Base run (canonical port) == identical to before."""
    c = SITE_RESET[site]
    final_state = c["state"]
    worker = ww.is_worker(site)
    auth_folder = (os.path.join(os.path.dirname(final_state), f"_wlogin_{site}_{ww.worker_port(site)}")
                   if worker else os.path.dirname(final_state))
    os.makedirs(auth_folder, exist_ok=True)
    url_env = dict(_URL_ENV)                                              # point auto_login at THIS container's URL
    _key = {"reddit": "REDDIT", "shopping": "SHOPPING", "shopping_admin": "SHOPPING_ADMIN", "gitlab": "GITLAB"}[site]
    url_env[_key] = f"http://localhost:{ww.worker_port(site)}" + ("/admin" if site == "shopping_admin" else "")
    env = dict(os.environ, **url_env, PLAYWRIGHT_BROWSERS_PATH=paths.PLAYWRIGHT_BROWSERS_PATH)
    log(f"[reset:{site}] regenerate storage_state (session-based; webarena auto_login) @ {url_env[_key]}")
    r = subprocess.run([_EVAL_PY, "-m", "browser_env.auto_login", "--site_list", site, "--auth_folder", auth_folder],
                       cwd=WEBARENA, env=env, capture_output=True, text=True, timeout=180)
    src = os.path.join(auth_folder, f"{site}_state.json")                # auto_login writes {'.'.join(comb)}_state.json
    if r.returncode != 0 or not os.path.exists(src):
        return False, ((r.stderr or "") + (r.stdout or ""))[-200:]
    if os.path.abspath(src) != os.path.abspath(final_state):
        shutil.copy(src, final_state)
    if worker:
        shutil.rmtree(auth_folder, ignore_errors=True)
    return True, ""


def _magento_worker_baseurl(site, name, port, log=print):
    """Magento WORKERS run the same *_configured snapshot (base_url baked to the BASE port, e.g. 7780) but on a
    DIFFERENT external port (e.g. 17780) -> /admin redirects to the base port and never renders -> reset's
    content-ready check times out. Re-point base_url at THIS worker's port after the container boots (wait for
    MySQL first). Only runs for Magento workers; base container (canonical port) already matches, so no-op."""
    mage = "/var/www/magento2/bin/magento"
    dbargs = ["mysql", "-u", "magentouser", "-pMyPassword", "magentodb", "-e"]
    for _ in range(18):                                                  # wait for the fresh container's MySQL
        if _docker("exec", name, *dbargs, "SELECT 1", timeout=20, retries=0).returncode == 0:  # this loop IS the retry
            break
        time.sleep(5)
    _docker("exec", name, mage, "setup:store-config:set", f"--base-url=http://localhost:{port}", timeout=150)
    _docker("exec", name, *dbargs,
            f'UPDATE core_config_data SET value="http://localhost:{port}/" WHERE path="web/secure/base_url";', timeout=30)
    _docker("exec", name, mage, "cache:flush", timeout=150)
    log(f"[reset:{site}] worker base_url -> http://localhost:{port}")


INJECT_SPEC_DEFAULT = os.path.join(ROOT, "experiments", "injected_post.json")


def _attach_thumbnail(site, sid, thumb_path, log=print):
    """Attach a local image file as submission `sid`'s thumbnail (Postmill/liip). Returns (uuid, media_type)
    to fold into the submission INSERT, or (None, 'url') on any failure. Postmill serves the listing thumb via
    the liip `submission_thumbnail_1x` filter (70x70 outbound) generated on demand from the ORIGINAL at
    /var/www/html/public/submission_images/<sha256hex>.jpg, with an `images` row (file_name + raw sha256)."""
    c = SITE_RESET[site]
    try:
        img = open(thumb_path, "rb").read()
    except Exception as e:
        log(f"[inject:{site}] thumbnail unreadable {thumb_path}: {e} -> text post")
        return None, "url"
    import hashlib
    import uuid as _uuid
    sha_hex = hashlib.sha256(img).hexdigest()
    fname = sha_hex + ".jpg"
    uid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "sge-inject-thumb/" + sha_hex))   # deterministic id (stable across resets)
    dest = "/var/www/html/public/submission_images/" + fname
    # 1) copy the original into the container's image store (idempotent) + fix ownership
    cp = subprocess.run(["docker", "exec", "-i", c["name"], "sh", "-lc",
                         f"cat > {dest} && chown www-data:www-data {dest} && echo OK"],
                        input=img, capture_output=True, timeout=60)
    if b"OK" not in (cp.stdout or b""):
        log(f"[inject:{site}] thumbnail copy failed: {(cp.stderr or b'').decode()[-160:]} -> text post")
        return None, "url"
    # 2) upsert the images row (bytea sha256 via decode hex). file_name is deterministic -> guard on it.
    w, h = 280, 280
    try:
        from PIL import Image as _I
        import io as _io
        w, h = _I.open(_io.BytesIO(img)).size
    except Exception:
        pass
    isql = (f"INSERT INTO images (id, file_name, width, height, sha256) "
            f"SELECT '{uid}', '{fname}', {w}, {h}, decode('{sha_hex}','hex') "
            f"WHERE NOT EXISTS (SELECT 1 FROM images WHERE file_name = '{fname}');")
    ri = subprocess.run(["docker", "exec", "-i", c["name"], "psql", "-U", "postmill", "-d", "postmill",
                         "-v", "ON_ERROR_STOP=1", "-c", isql], capture_output=True, text=True, timeout=60)
    if ri.returncode != 0:
        log(f"[inject:{site}] images row failed: {((ri.stderr or '')+(ri.stdout or '')).strip()[-160:]} -> text post")
        return None, "url"
    # id actually stored for this file_name (handles the pre-existing-row case)
    got = subprocess.run(["docker", "exec", "-i", c["name"], "psql", "-U", "postmill", "-d", "postmill",
                          "-tAc", f"SELECT id FROM images WHERE file_name='{fname}';"],
                         capture_output=True, text=True, timeout=30)
    real_uid = (got.stdout or "").strip() or uid
    return real_uid, "image"


def _prewarm_thumb(site, fname, log=print):
    """Force liip to generate + cache the 70x70 thumbnail now, so the agent's first paint of the listing
    doesn't race on-demand generation (a broken/empty thumb in the screenshot)."""
    c = SITE_RESET[site]
    # The liip RESOLVE route generates + caches the thumbnail (302 -> cache file); the plain cache URL just
    # 404s until it exists. Warm both filter sets the listing may request (1x and 2x).
    for flt in ("submission_thumbnail_1x", "submission_thumbnail_2x"):
        url = f"http://localhost/media/cache/resolve/{flt}/{fname}"
        subprocess.run(["docker", "exec", c["name"], "sh", "-lc",
                        f"curl -s -o /dev/null '{url}' || wget -q -O /dev/null '{url}' || true"],
                       capture_output=True, timeout=30)


def _inject_post(site, log=print):
    """EXPERIMENT HOOK (opt-in, default OFF): re-insert a custom submission after a reset.

    A reset destroys the container and recreates it from the image, so the DB reverts to the baked
    snapshot — anything written into a running container is gone before the next task starts. This
    re-applies one submission after every reset so EVERY run sees it. reddit/Postmill only.

    Enable with WA_INJECT_POST=1; spec from WA_INJECT_POST_FILE (default experiments/injected_post.json).
    Postmill's "new" sort orders by submission id DESC, so the spec's `id` must exceed the forum's max id
    (the shipped snapshot's global max is 137402) for the post to land at the TOP of /f/<forum>/new.
    Idempotent: deletes that id first. Failures are logged, never fatal — a broken spec must not
    silently turn into a reset failure that skips tasks."""
    if site != "reddit":
        return False
    spec_path = os.getenv("WA_INJECT_POST_FILE") or INJECT_SPEC_DEFAULT
    try:
        spec = json.load(open(spec_path, encoding="utf-8"))
    except Exception as e:
        log(f"[inject:{site}] cannot read spec {spec_path}: {e} -> skipped")
        return False
    sid, forum = int(spec["id"]), spec.get("forum", "books")
    author = spec.get("author", "BitterStatus9")
    title, body = spec.get("title", ""), spec.get("body", "")
    ts = spec.get("timestamp", "2023-04-01 12:00:00+00")
    # Optional thumbnail: attach a local image so the post shows a 70x70 badge on the listing. Path from the
    # spec ("thumbnail", repo-relative or absolute); done BEFORE the submission insert so image_id can be set.
    img_uid, media_type, thumb_fname = None, "url", None
    thumb = spec.get("thumbnail")
    if thumb:
        thumb_abs = thumb if os.path.isabs(thumb) else os.path.join(ROOT, thumb)
        img_uid, media_type = _attach_thumbnail(site, sid, thumb_abs, log=log)
        if img_uid:
            import hashlib
            thumb_fname = hashlib.sha256(open(thumb_abs, "rb").read()).hexdigest() + ".jpg"
    # Dollar-quote title/body so ANY content (quotes, backslashes, newlines, injected instructions) is
    # inserted verbatim with no escaping. Pick a tag that cannot appear inside the payload.
    tag = "SGEQ"
    while f"${tag}$" in title or f"${tag}$" in body:
        tag += "Q"
    img_col = f"'{img_uid}'::uuid" if img_uid else "NULL"
    sticky = "true" if spec.get("sticky") else "false"          # pin to top of the forum (green "pinned" title)
    sql = f"""
BEGIN;
DELETE FROM submission_votes WHERE submission_id = {sid};
DELETE FROM submissions WHERE id = {sid};
INSERT INTO submissions
  (id, forum_id, user_id, title, timestamp, url, body, image_id, sticky, ranking, moderated,
   user_flag, locked, last_active, comment_count, net_score, visibility, media_type)
SELECT {sid}, f.id, u.id, ${tag}${title}${tag}$, '{ts}'::timestamptz, NULL, ${tag}${body}${tag}$, {img_col},
       {sticky}, {int(spec.get('ranking', 1680350000))}, false,
       '{spec.get("user_flag", "t3_1282f4k")}', false, '{ts}'::timestamptz, 0,
       {int(spec.get('net_score', 0))}, 'visible', '{media_type}'
FROM forums f, users u WHERE f.name = '{forum}' AND u.username = '{author}';
COMMIT;
"""
    c = SITE_RESET[site]
    r = subprocess.run(["docker", "exec", "-i", c["name"], "psql", "-U", "postmill", "-d", "postmill",
                        "-v", "ON_ERROR_STOP=1", "-f", "-"],
                       input=sql, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or "INSERT 0 1" not in (r.stdout or ""):
        log(f"[inject:{site}] FAILED: {((r.stderr or '') + (r.stdout or '')).strip()[-200:]}")
        return False
    if thumb_fname:
        _prewarm_thumb(site, thumb_fname, log=log)
    log(f"[inject:{site}] post id={sid} -> /f/{forum}/new (top): {title[:60]!r}"
        + (f"  +thumb({media_type})" if img_uid else ""))
    return True


def reset(site="reddit", timeout=300, poll=2.0, want_login=True, log=print):
    """Destroy + recreate the site's container; block until pristine + ready (+ re-login for session sites)."""
    c = SITE_RESET[site]
    timeout = c.get("timeout", timeout)                                  # gitlab needs longer to boot
    t0 = time.time()
    if not _daemon_ok():
        return {"ok": False, "error": "docker daemon not responding (socket hang / busy)",
                "seconds": round(time.time() - t0, 1)}
    deadline = t0 + timeout                                              # bound the WHOLE reset (rm/run INCLUDED) by `timeout`
    hostports = [c["site_port"]] + ([c["ctrl_port"]] if c["ctrl_port"] else [])
    port_args = (["-p", f"{c['site_port']}:{c.get('internal_port', '80')}"]
                 + (["-p", f"{c['ctrl_port']}:8877"] if c["ctrl_port"] else []))
    cid, err = None, ""
    for attempt in range(3):                                            # rm -> wait ports free -> run; the LOOP is the retry
        if deadline - time.time() < 15:                                 # no budget left for another create attempt
            err = err or "reset timed out before the container came up"
            break
        log(f"[reset:{site}] docker rm -f {c['name']}" + (f" (retry {attempt})" if attempt else ""))
        rm = _docker("rm", "-f", c["name"], timeout=min(60, int(deadline - time.time())), retries=1)
        if rm.returncode != 0 and "no such container" not in (rm.stderr or "").lower():
            log(f"[reset:{site}] rm warn: {rm.stderr.strip()[:160]}")
        if not _wait_ports_free(hostports, timeout=min(25, max(2, int(deadline - time.time())))):
            log(f"[reset:{site}] ports {hostports} still busy — trying anyway")
        log(f"[reset:{site}] docker run fresh {c['image']}")
        run = _docker("run", "-d", "--name", c["name"], *port_args, c["image"], *c.get("run_extra", []),
                      timeout=min(90, max(15, int(deadline - time.time()))), retries=1)
        if run.returncode == 0:
            cid = (run.stdout or "").strip()[:12]
            break
        err = (run.stderr or "").strip()
        log(f"[reset:{site}] docker run failed (attempt {attempt}): {err[:200]}")
        time.sleep(2.0 * (attempt + 1))                                 # let the port/name settle, then retry
    if cid is None:
        return {"ok": False, "error": f"docker run failed / timed out: {err[:200]}",
                "seconds": round(time.time() - t0, 1)}
    if site in ("shopping", "shopping_admin") and ww.is_worker(site):    # Magento worker: match base_url to this port
        _magento_worker_baseurl(site, c["name"], c["site_port"], log=log)
    # For relogin sites the saved session is dead post-swap -> verify CONTENT first (skip login), regen, then re-check.
    content_login = want_login and not c["relogin"]
    detail = {}
    while time.time() - t0 < timeout:
        ready, detail = check_ready(site, want_login=content_login)
        if ready:
            break
        time.sleep(poll)
    else:
        return {"ok": False, "error": "timeout waiting for content-ready", "seconds": round(time.time() - t0, 1), "detail": detail}
    if c["relogin"] and want_login:
        ok, err = _regen_login(site, log=log)
        if not ok:
            return {"ok": False, "error": f"relogin failed: {err}", "seconds": round(time.time() - t0, 1), "detail": detail}
        _, detail = check_ready(site, want_login=True)                   # informational: confirm the fresh session authes
    if os.getenv("WA_INJECT_POST"):                                      # opt-in experiment hook (default OFF)
        _inject_post(site, log=log)
    secs = round(time.time() - t0, 1)
    log(f"[reset:{site}] READY in {secs}s  {detail}")
    return {"ok": True, "seconds": secs, "cid": cid, "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="reddit", choices=list(SITE_RESET))
    ap.add_argument("--check", action="store_true", help="readiness/login check only (no reset)")
    ap.add_argument("--no-login", action="store_true", help="skip the login re-auth assertion")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    if a.check:
        ready, detail = check_ready(a.site, want_login=not a.no_login)
        print(json.dumps({"site": a.site, "ready": ready, **detail}, indent=2))
        return
    rep = reset(a.site, timeout=a.timeout, want_login=not a.no_login)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
