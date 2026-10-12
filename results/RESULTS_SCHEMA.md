# Results layout

New runs land in `results/runs/<task>_<mode>/` (WebArena) or `results/runs/camel_webarena/<qtag>/<task>_camel/`
(CaMeL). Our shipped numbers are under `results/published/` (JSON + logs only — screenshots stripped).

## Per-task directory
**Secure / no-query** (`<task>_secure`, `<task>_noquery`):
`result.json`, `meta.json`, `runtime.log`, `model_response.txt`, `final_url.json`, `op_trace.json`; per turn
`turn_N_vtop.png` (trusted V⊤), `turn_N_after.png`, `turn_N_a11y.txt`, `turn_N_op.json`; on `query` turns the
isolated `turn_N_mark<K>_vg.png` + a11y; memory dumps `K_turn_N.json` / `G_turn_N.json`. *(shipped copies omit the pngs.)*

**Baseline** (`<task>_baseline`): `result.json`, `meta.json`, `runtime.log`, `model_response.txt`, `final_url.json`;
per step `step_N.png`, `step_N.axtree.txt`, `step_N.response.txt`.

## `result.json` keys
- common: `task`, `webarena_verdict` (`SUCCESS`/`FAILURE`), `webarena_passed`, `turns`, `final_url`, `metrics`
- secure/no-query also: `end_status`, `end_answer`
- `metrics`: secure = `{actuator:{calls,input,output,thinking,time}, shadow:{…}, task_time}`;
  baseline = `{baseline:{…}, task_time}` (per-component tokens + wall-time)

## `meta.json` keys
`task, goal, start_url, run_id, model, provider, shadow_provider, mode, env, max_steps, screen_size, a11y, coord_frame`

Suite roll-ups: `results/runs/<...>/_suite/suite_<mode>_<timestamp>.json` (consumed by `compare_arms.py`).
