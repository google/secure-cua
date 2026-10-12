# secure-gui-eval — masking defense for GUI agents, evaluated

> [!NOTE]
> This project is intended for demonstration purposes only. It is not
> intended for use in a production environment.

A reproducible artifact for our study of **prompt-injection defenses for computer-use (GUI) agents**. It runs
**four agent arms** on **two benchmarks** with identical tasks and the same official scorers, so the arms are
directly comparable.

| arm | what it is |
|---|---|
| **SECURE** | our masking defense — the agent sees a masked screenshot (V⊤); a `query`→`compute`→`act` loop reads one untrusted region at a time through an isolated, type-checked channel |
| **BASELINE** | an unrestricted computer-use agent (no masking) — the utility control |
| **CaMeL-CUA** | the dual-LLM plan-and-execute defense (blind planner + grounder VLM + quarantined Q-LLM) |
| **NO-QUERY** | ablation: masking with the read channel removed (`act`+`return` only) — isolates what `query` buys |

**Benchmarks:** **WebArena** (reddit / shopping / shopping_admin / gitlab — plain Docker, portable) and
**REAL / omnizon** (agisdk) — both run through the **same headed-Chromium/Playwright pipeline**; the only
difference is the site source (a local container for WebArena vs a public hosted URL for REAL) and the scorer.

---

## Repository layout
```
secure-gui-eval/
├── README.md            ← setup · run · analyze
├── .env.example         ← the ONE file you edit: model-provider credentials
├── env.sh               ← auto runtime env (no editing; sourced by every script)
├── setup.sh · setup/    ← one-shot installer + individual steps + verify.sh
├── secure_bench/        ← all our Python (paths auto-resolve from the repo — no config)
├── camel/               ← vendored CaMeL (our modified copy)
├── extension/           ← the masking browser extension
├── scripts/             ← the run commands, one per arm × benchmark
├── results/             ← published/ (our numbers) · runs/ (YOUR runs) · RESULTS_SCHEMA.md
└── fetched/             ← created by setup: WebArena harness clone + browsers + agisdk env + auth states
```
Every location derives from the repo path (`secure_bench/paths.py`) — **no path configuration**. You provide only
credentials (`.env`) and, optionally, a relocated data dir (`SGE_FETCHED`, §2).

---

## Quickstart
```bash
conda create -y -n sge python=3.12 && conda activate sge   # the primary env (Python 3.10–3.12; not 3.13)
bash setup.sh                            # install deps into it + WebArena docker + extension, then a preflight
cp .env.example .env && nano .env        # add model-provider credentials
gcloud auth application-default login    # if using Claude/Gemini via Vertex
bash setup/verify.sh                     # preflight — should print "READY TO RUN"
# one task, each arm (with the `sge` env activated):
scripts/webarena_baseline.sh --workers 1 --model claude-opus-4-8 --tasks reddit-404
scripts/webarena_secure.sh   --workers 1 --model claude-opus-4-8 --tasks reddit-404
scripts/webarena_noquery.sh  --workers 1 --model claude-opus-4-8 --tasks reddit-404
scripts/webarena_camel.sh    --workers 1 --arms opus             --tasks reddit-404
```

---

## 1. Requirements

- **Host tools:** **conda** (Miniconda/Anaconda — it supplies the Python **3.10–3.12** primary env) · Docker (rootless
  or system) · `git` · Node 18+ · Xvfb. `setup/00_prereqs.sh` installs the non-conda ones (assumes root/sudo).
  **Use Python 3.10–3.12, not 3.13** — `numpy<2` (required by the WebArena harness) has no 3.13 wheels; `conda create
  -n sge python=3.12` gives you a clean one regardless of your system Python.
- **Realbench (REAL/omnizon):** runs against a **public hosted site**, so it needs only outbound network + the
  agisdk scorer env (`SETUP_REALBENCH=1`, built by `setup/10`).
- **Credentials** (≥1 provider, set in `.env` — §3): **Claude & Gemini via Google Vertex** (a GCP project +
  `gcloud auth application-default login`); **Azure OpenAI** (CaMeL's GPT-5 planner, the GPT arm, the WebArena
  fuzzy-match judge). Direct Anthropic/Gemini/OpenAI keys also work.
- **Disk:** WebArena images are large — ~194 GB of tarballs (shopping 63 · gitlab 72 · reddit 50 · shopping_admin 9);
  `docker load` roughly doubles it, so budget **~400 GB**. You only need the site(s) you run (reddit-only ≈ 50 GB).
  Tarballs land in `$SGE_FETCHED/wa_tarballs`; delete a tarball once its image is loaded.

---

## 2. Set up the environment

The whole project installs from **one `requirements.txt`** into **one conda env** — the four arms **plus** the official
WebArena scorer, merged. Realbench adds one tiny separate `agisdk` scorer env. Conda supplies the interpreter, `pip`
installs the deps — no `uv`, no build backend.

### Automated
```bash
conda create -y -n sge python=3.12 && conda activate sge   # create + activate the primary env FIRST
bash setup.sh                     # installs deps into the active env + WebArena docker + extension, then a preflight
SETUP_REALBENCH=1 bash setup.sh   # also build the agisdk scorer env (needed for realbench)
```

### Manual (exactly what `setup.sh` does — run from the repo root, one shell session)
```bash
# 0. fetched-DATA location — the only path you set (absolute). Keep it on a big LOCAL disk.
export SGE_FETCHED="$PWD/fetched"                        # or e.g. /scratch/$USER/sge
export PLAYWRIGHT_BROWSERS_PATH="$SGE_FETCHED/.pw-browsers"; mkdir -p "$SGE_FETCHED"

# 1. host tools (conda, git, node, xvfb, docker)  — or: bash setup/00_prereqs.sh   (python comes from conda, step 3)

# 2. clone the WebArena eval harness AT THE PINNED COMMIT (env.sh puts it on PYTHONPATH).
#    The checkout matters — a drifting HEAD changes config_files/generate_test_data and breaks scoring.
git clone https://github.com/web-arena-x/webarena "$SGE_FETCHED/webarena_official"
git -C "$SGE_FETCHED/webarena_official" checkout dce04686a56253aefba7b18a4fa0937cf1dc987b

# 3. create + ACTIVATE the primary conda env (Python 3.10–3.12; numpy<2 has no 3.13 wheels). Keep it ACTIVE through
#    step 8 — setup/20 + verify.sh use it (env.sh reads the active conda env). Conda gives the interpreter, pip the deps.
conda create -y -n sge python=3.12 && conda activate sge
pip install --upgrade pip
pip install -r requirements.txt      # applies the pins (numpy<2, scipy>=1.11, anthropic==0.64.0, pydantic-ai==0.8.1, …)
playwright install chromium          # → $PLAYWRIGHT_BROWSERS_PATH (set in step 0; must be the SAME shell)

# 4. (realbench only) the separate agisdk scorer env — agisdk needs gymnasium 1.x + numpy 2.x, so it can't share
#    the primary env (numpy<2).
#    A SEPARATE conda env at a fixed path, invoked only as a subprocess by path (never activated):
conda create -y -p "$SGE_FETCHED/.agisdk_venv" python=3.11
"$SGE_FETCHED/.agisdk_venv/bin/pip" install --upgrade pip
"$SGE_FETCHED/.agisdk_venv/bin/pip" install agisdk==0.3.5
"$SGE_FETCHED/.agisdk_venv/bin/python" -m playwright install chromium

# 5. build the masking extension
( cd extension/mask_extension && node build_seed.js )

# 6. stand up the WebArena site containers (heavy download + config; scripted)
bash setup/20_webarena_docker.sh
#    (realbench doesn't use these containers — its site is a public hosted URL; it needs only the step-4 scorer env)

# 7. credentials, then verify
cp .env.example .env       # then EDIT .env: fill ONE provider (see §3). For Vertex also: gcloud auth application-default login
bash setup/verify.sh       # → ">>> READY TO RUN"
```

**Dependencies & pins.** `requirements.txt` installs *only* third-party packages — the bundled code (`secure_bench`,
vendored `camel`, the WebArena harness) runs straight from `PYTHONPATH`, so there's **no build backend** and
it works on a **private-index-only** machine (it uses whatever index your pip is configured for). Most versions float,
but a few are load-bearing and **must stay pinned**:
- `numpy<2` + `scipy>=1.11` — the WebArena harness fails to import otherwise (`Cannot interpret 'ScalarT'`, or `np.long`).
- `anthropic==0.64.0` + `pydantic-ai==0.8.1` + `agentdojo==0.1.34` — CaMeL's grounder path breaks on newer versions
  (the grounder silently returns "not found" and CaMeL grades unfairly low).

A fresh env from `requirements.txt` resolves to these automatically. If a run dies with `ModuleNotFoundError: X`, that
package just isn't on your index — `pip install X` and add it. **One non-pip download:** `playwright install chromium`
fetches from Playwright's CDN (not your index) — needs outbound net or `PLAYWRIGHT_DOWNLOAD_HOST`; else supply Chromium
and point `PLAYWRIGHT_BROWSERS_PATH`/`CHROMIUM_EXE` at it.

**Off the code drive** (quota'd / read-only root): conda envs already live outside the repo (in conda's env dir), and
`SGE_FETCHED` points the fetched **data** at local disk (set it in `.env`). `env.sh` reads the **active** conda env
(`$CONDA_PREFIX`) — just `conda activate sge` in each shell. Move only the browsers with `PLAYWRIGHT_BROWSERS_PATH`.
Code stays in the repo.

| step · script | creates |
|---|---|
| host tools · `setup/00_prereqs.sh` | conda / docker / git / node / xvfb (missing ones; `CHECK_ONLY=1` to only report) |
| **primary env** · `setup/10_venvs.sh` | the 4 arms **+** WebArena scorer, one conda env |
| agisdk env · `setup/10_venvs.sh` | realbench scorer only (`SETUP_REALBENCH=1`; separate conda env — agisdk's gymnasium 1.x + numpy 2.x vs the primary's numpy<2) |
| extension · `setup/40_extension.sh` | `mask_extension/seed.js` |
| WebArena docker · `setup/20_webarena_docker.sh` | 4 site images + task configs + auth states |

---

## 3. Credentials
```bash
cp .env.example .env         # then edit — fill ONE provider path (the file is commented)
# Vertex (recommended): set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI=1,
#                       AZURE_OPENAI_* (GPT-5 planner + fuzzy-match judge), then:
gcloud auth application-default login    # set GOOGLE_APPLICATION_CREDENTIALS to the json it prints
bash setup/verify.sh                     # PASS/FAIL per component, then "READY TO RUN"
```
> Vertex ADC tokens expire every few hours — re-run the `gcloud` line when a run fails with `Reauthentication is needed`.

---

## 4. Run experiments

**Activate the primary env first** (`conda activate sge`) — the scripts source `env.sh`, which then uses the active
conda env. Pick the model with `--model`, tasks with `--tasks`, parallel site-containers with `--workers`. **All output
→ `results/runs/`.**

### WebArena  (`--model` ∈ `claude-opus-4-8`, `gpt-5`, `gemini-3.1-pro-preview`)
```bash
scripts/webarena_baseline.sh --workers 1 --model claude-opus-4-8 --tasks reddit-404 shopping-21
scripts/webarena_secure.sh   --workers 3 --model claude-opus-4-8 --tasks reddit-404 shopping-21 gitlab-102
scripts/webarena_noquery.sh  --workers 1 --model claude-opus-4-8 --tasks reddit-404
# CaMeL: --arms picks the quarantined Q-LLM (opus/gemini/gpt); planner=GPT-5, grounder=Sonnet (fixed)
scripts/webarena_camel.sh    --workers 3 --arms opus gemini gpt  --tasks reddit-404 shopping-21
```
- `--skip-done` **resumes** (skips tasks/seeds that already have a result).
- **Whole site:** `--site {reddit|shopping|shopping_admin|gitlab}` runs every single-site task for it (~100+; use
  `--skip-done`). Mutually exclusive with `--tasks`. (`python secure_bench/wa_site_tasks.py <site>` lists the ids.)
- **All three masking-family arms at once:** `scripts/run_all.sh webarena --workers 3 --model claude-opus-4-8 --tasks reddit-404`

### REAL / omnizon  (agisdk-scored; needs the scorer env, `SETUP_REALBENCH=1`)
Same flags as the WebArena arms — only the task ids differ (`omnizon-<N>`):
```bash
scripts/realbench_baseline.sh --model claude-opus-4-8 --tasks omnizon-3 omnizon-9
scripts/realbench_secure.sh   --model claude-opus-4-8 --tasks omnizon-3 omnizon-9
scripts/realbench_noquery.sh  --model gpt-5           --tasks omnizon-3
scripts/realbench_camel.sh    --workers 2 --arms opus gemini gpt --tasks omnizon-3 omnizon-9
```

### Fair-comparison knob
- **`--max-hist N`** — cap the LLM history to the last N turns (secure / baseline / no-query). Defaults without it:
  **secure/no-query = full**, **baseline = 3**. CaMeL is plan-and-execute, so it has no history window (N/A). Equalizes
  the history window across arms so a utility gap reflects the defense, not the context budget.
  ```bash
  scripts/webarena_secure.sh  --workers 1 --model claude-opus-4-8 --tasks reddit-404 --max-hist 5
  ```

### Multiple seeds → pass@k
Run each task across **K independent repeats** (the LLMs sample stochastically) and report **unbiased pass@k**
(Chen et al.: `1 − C(n−c,k)/C(n,k)`, averaged over tasks). `--seeds K` on any arm (both benchmarks):
```bash
scripts/webarena_secure.sh  --model claude-opus-4-8 --site reddit --seeds 5 --skip-done
scripts/realbench_secure.sh --model claude-opus-4-8 --tasks omnizon-3 omnizon-9 --seeds 5
python secure_bench/passk.py results/runs --by-site        # the pass@k table (+ optional --csv/--json)
```
Each seed writes `<task>_<arm>/seed<k>/result.json`; `--skip-done` is **per-seed**. `seeds=1` (the default, i.e.
omitting the flag) keeps the original flat single-run layout.

---

## 5. Where results are stored

Every run writes to **`results/runs/`**, one directory per task+arm (`<task>_<arm>/`; CaMeL:
`camel_webarena/q<arm>/<task>_camel/`; realbench: `<task>_v2_..._<arm>/`; multi-seed adds `seed<k>/`).
```
results/
├── published/           ← OUR shipped numbers (JSON + logs)
├── runs/                ← YOUR runs land here
└── RESULTS_SCHEMA.md    ← full field-by-field description
```
Each dir has **`result.json`** (`webarena_verdict` SUCCESS/FAILURE · `turns` · `final_url` · `metrics` = per-component
tokens + wall-time), **`meta.json`**, **`runtime.log`**, and per-turn artifacts — secure: `turn_N_vtop.png`,
`turn_N_mark<K>_vg.png` (isolated V_g on `query` turns), `K_turn_N.json`/`G_turn_N.json`; baseline: `step_N.png`.
```bash
cat results/runs/reddit-404_secure/result.json | python3 -m json.tool | grep -E 'verdict|turns|final_url'
```

---

## 6. Analyze results
With the primary env **activated** (`conda activate sge`):
```bash
source env.sh    # PYTHONPATH + creds (uses the active conda env)
python secure_bench/passk.py results/runs --by-site        # unbiased pass@k, per arm & site (all arms, both benchmarks)
python secure_bench/collate_metrics.py results/published   # pass rates + token/USD → metrics_summary.{csv,json}
python secure_bench/compare_arms.py --out-root results/runs # baseline vs secure vs no-query + secure op-mix
VIEWER_ROOT=results/published python secure_bench/secure_viewer.py   # trajectory viewer @ :8899 (+ PDF export)
```

---

## Reference

**Headline results** (`results/published/`):
- **Masking is near-free on utility** — SECURE matches BASELINE on WebArena (~88% of baseline-passable tasks preserved).
- **`query` is necessary** — NO-QUERY → 0 on the payload tasks BASELINE/SECURE pass (can't recover the masked answer).
- **CaMeL-CUA** far below masking on the same tasks.

**File map (`secure_bench/`):**
| files | role |
|---|---|
| `run_secure_webarena.py` · `secure_{actuator,shadow,store,protocol,diff,summary}.py` `smask_runtime.py` | **SECURE** (both benchmarks) |
| `run_baseline_webarena.py` | **BASELINE** (both benchmarks) |
| `run_noquery_webarena.py` `noquery_actuator.py` `noquery_tool_spec.py` | **NO-QUERY** (both benchmarks) |
| `run_camel_orchestrator.py` (+ vendored `camel/`) | **CaMeL-CUA** |
| `run_suite.py` `run_parallel.py` · `wa_reset/wa_workers/wa_a11y/wa_official_eval/wa_camel_helpers.py` `agisdk_eval.py` | shared harness, parallelism, official scorers |
| `llm_adapters.py` `tool_spec.py` `metrics.py` `paths.py` `benchmarks.py` | shared core (providers, ops, cost, path resolution, per-benchmark seams) |
| `passk.py` `secure_viewer.py` `report_pdf.py` `collate_metrics.py` `collate_by_site.py` `compare_arms.py` `wa_site_tasks.py` | analysis (pass@k, viewer, cost) + whole-site task lister |

Scoring is the **official** WebArena harness (`wa_official_eval.py`) and the agisdk REAL evaluator (`agisdk_eval.py`) —
identical across all four arms.

## Google Open Source Software Vulnerability Rewards Program

 Eligibility for the [Google Open Source Software Vulnerability Rewards
  Program](https://bughunters.google.com/open-source-security) is determined by the [Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

