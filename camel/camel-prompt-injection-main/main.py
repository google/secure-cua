# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import os
import re

import cyclopts
from agentdojo import attacks, benchmark, logging
from agentdojo.task_suite import get_suite
from openai.types.chat import ChatCompletionReasoningEffort

from camel.interpreter.interpreter import MetadataEvalMode
from camel.models import make_tools_pipeline

# The realbench (REAL/omnizon) + webarena task suites are built dynamically below from
# camel_cua.ui_benchmark.suites.realbench; only webarena + omnizon tasks are registered.

def main(
    model: str,
    use_original: bool = False,
    reasoning_effort: ChatCompletionReasoningEffort = "medium",
    thinking_budget_tokens: int | None = None,
    ad_defense: str | None = None,
    run_attack: bool = False,
    replay_with_policies: bool = False,
    suites: list[str] | None = None,
    eval_mode: MetadataEvalMode = MetadataEvalMode.NORMAL,
    q_llm: str | None = None,
    q_llm_second_check: str | None = None,
    user_tasks: list[str] | None = None,
    model_defense_level: int = 0,
    system_defense_level: int = 0,
    max_attempts: int = 10,
    pllm_exposure_level: int = 0,
    path_to_vlm: str | None = None,
    path_to_vm: str | None = None,
    port: int = 15001,
    unoptimized_cua: bool = False,
    plan_analysis_only: bool = False,
):
    """Example usage of the defense.


    Other newer models might work as well.

    Args:
        model: the model to use. it should be {provider}:model_name. For example, "google:gemini-2.5-pro-preview-06-05".
        use_original: whether to use the original model with tool calling API instead of CaMeL
        reasoning_effort: for OpenAI reasoning models. How much the model should reason. Can be "low", "medium", "high".
        thinking_budget_tokens: how many tokens Anthropic reasoning models can use. Note that Anthropic reasoning models are not supported yet.
        ad_defense: whether to use a defense from AgentDojo and which one. It must be used in conjunction with `--use-original`.
            Tested defenses are "tool_filter", "repeat_user_prompt", "spotlight_with_delimiting"
        run_attack: whether to run the attack (it uses AgentDojo's `important_instructions` attack)
        replay_with_policies: replay the run with the given model enforcing security policies. Note that the equivalent run (with same model and attack config)
            should have already been run.
        suites: which suites to run AgentDojo on (can be a list from `["workspace", "banking", "travel", "slack"]`)
        eval_mode: which eval mode to use when propagating dependencies.
        q_llm: what model to use as a quarantined llm. If None, the same as `model` is used.
        q_llm_second_check: what model to use as a secondary check quarantined llm (will be used for model_level_defense >= 3). If None, the same as `model` is used.
        model_defense_level: the level of defense to use for the model (between 0 no defense to 4 first checker with screenshot and dom and second checker with only screenshot). Only used for realbench tasks.
        system_defense_level: the function leniency (between 0 all funcions to 2 only find + UI functions). Only used for realbench tasks.
        max_attempts: maximum number of retry attempts for code generation and execution. The planner will retry up to this many times 
            if the code has interpreter errors or (for UIEnv tasks) if the task is not completed successfully. Defaults to 10.
        pllm_exposure_level: controls the level of feedback provided to the planner between retry attempts (only affects UIEnv tasks). 
            Level 0: no feedback is provided between retries. Level 1: provides the failed code and a simple failure message. 
            Level 2: provides the failed code with annotations (print outputs inline) and detailed failure message with debugging hints. 
            Defaults to 0. Has no effect on non-UIEnv tasks.
        path_to_vlm: path to the vision language model to use for the realbench tasks. Need to give an absolute path.
        path_to_vm: path to the vision model to use for the realbench tasks. Need to give an absolute path.
        port: port number to use for the browser connection in realbench tasks. Defaults to 15001.
        unoptimized_cua: if True, run CAMEL+CUA with a baseline system prompt (no CUA-specific
            planning guidance) and remove the verify_hypothesis tool. Only affects the UITARS
            backend in the realbench suite; ignored otherwise. Use this for an apples-to-apples
            comparison against the optimized CAMEL+CUA setup.
        plan_analysis_only: if True, only generates plans without executing them.
            Plans are saved to disk for offline analysis. No tool calls are made
            against the environment. The number of plans per task is controlled
            by max_attempts.
    """
    attack_name = "important_instructions"
    p = (path_to_vlm or "").lower()
    is_uitars, is_opencua, is_claude = ("tars" in p), ("opencua" in p), ("claude" in p)
    is_gemini, is_openai = ("gemini" in p or "google" in p), ("gpt" in p or "openai" in p)
    # --unoptimized-cua only meaningful for UITARS backend; treat as no-op otherwise.
    unoptimized_cua_effective = unoptimized_cua and is_uitars


    suites = suites or ["workspace", "banking", "travel", "slack", "realbench"]
    total_utility_results = []
    total_security_results = []
    logdir = Path("./logs_agentdojo")

    for suite_name in suites:
        part_path = logdir #/ f"{model.split(':')[1]}+camel" / suite_name
        if plan_analysis_only and suite_name != "realbench":
            part_path = part_path / "plan_only"
        if suite_name in ("realbench", "webarena"):
            logdir = Path(os.environ.get("CAMEL_LOG_DIR", "./logs_realbench"))
            prefix = next((k for k, b in (("uitars", is_uitars), ("opencua", is_opencua), ("claude", is_claude), ("gemini", is_gemini), ("openai", is_openai)) if b), "unknown")
            part_path = (
                logdir
                / prefix
                / f"{model.split(':')[1].split('/')[1] if '/' in model else model.split(':')[1]}+camel"
                / f"modeldef_{model_defense_level}_systdef_{system_defense_level}"
                / f"qllm1_{q_llm.split(':')[1].split('/')[1] if '/' in q_llm else q_llm.split(':')[1]}_qllm2_{'None' if q_llm_second_check is None else (q_llm_second_check.split(':')[1].split('/')[1] if '/' in q_llm_second_check else q_llm_second_check.split(':')[1])}"
            )
            if unoptimized_cua_effective:
                part_path = part_path / "unoptimized"
            if plan_analysis_only:
                part_path = part_path / "plan_only"

        tools_pipeline = make_tools_pipeline(
            model,  # type: ignore
            use_original,
            replay_with_policies,
            attack_name,
            reasoning_effort,
            thinking_budget_tokens,
            ("realbench" if suite_name in ("realbench", "webarena") else suite_name),   # webarena reuses the realbench policy/prompt pipeline
            ad_defense,
            eval_mode,
            q_llm,  # type: ignore, not used in the pipeline
            part_path = part_path,
            max_attempts=max_attempts,
            pllm_exposure_level=pllm_exposure_level,
            system_defense_level=system_defense_level,
            unoptimized_cua=unoptimized_cua_effective,
            plan_analysis_only=plan_analysis_only,
        )
        if suite_name in ("realbench", "webarena"):
            if q_llm is not None:
                assert q_llm_second_check != q_llm, "q_llm_second_check should be different from q_llm"
            if model_defense_level >= 3:
                assert q_llm_second_check is not None, "For model_defense_level >= 3, q_llm_second_check, should be provided"
            from camel_cua.ui_benchmark.suites import realbench
            if suite_name == "webarena":
                # WebArena backend: CaMeL on a Playwright/CDP container browser, scored by the OFFICIAL
                # WebArena evaluator (CaMeL is UNMASKED — it is its own defense). Register only the requested
                # ids (bare, e.g. reddit-27), else the full 40-task set.
                suite = realbench.make_task_suite_WebArena(
                        system_defense_level=system_defense_level
                    )
                from camel_cua.webarena_tasks import register_webarena_tasks
                from wa_camel_helpers import TASK_IDS as _WA_ALL
                _wa_ids = [t.split("/")[-1] for t in user_tasks] if user_tasks else _WA_ALL
                cli_to_suite_id = register_webarena_tasks(suite, _wa_ids)
            else:                                # the "realbench" suite = the REAL/omnizon path (Anthropic-backed Playwright)
                suite = realbench.make_task_suite_Anthropic(
                        system_defense_level=system_defense_level
                    )
                
            if suite_name != "webarena":
                cli_to_suite_id = {}                 # legacy desktop JSON tasks removed; register only REAL/omnizon below

                # Register the requested REAL-benchmark tasks (ANY agisdk site — omnizon/udriver/staynb/...),
                # agisdk-scored, and merge their CLI names so --user-tasks <site>-N resolves. Registered at
                # indices 9000+ (no collisions with other registered tasks). Defaults to the 10 omnizon ids
                # when no REAL-pattern task is requested (back-compat). REAL ids look like "<site>-<number>".
                from camel_cua.omnizon_tasks import register_omnizon_tasks
                _rb = [t.split("/")[-1] for t in (user_tasks or [])
                       if "-" in t.split("/")[-1] and t.rsplit("-", 1)[-1].isdigit()
                       and not t.split("/")[-1].startswith("user_task")]
                cli_to_suite_id.update(register_omnizon_tasks(suite, task_ids=(_rb or None)))

            # Map CLI names like "chrome/<uuid>" -> "user_task_<n>"
            mapped = None
            if user_tasks:
                mapped = [cli_to_suite_id.get(t, t) for t in user_tasks]
                # Optional: assert they all mapped
                missing = [t for t, m in zip(user_tasks, mapped) if t == m]
                if missing:
                    print("Warning: these task names didn’t map:", missing)
            # pass mapped (or None) to the benchmark fn
            selected = mapped
            for elem in selected:
                for usertask in suite._user_tasks[elem].values():
                    usertask.model_defense_level = model_defense_level
                    usertask.q_llm = q_llm if q_llm is not None else model
                    usertask.q_llm_second_check = q_llm_second_check if q_llm_second_check is not None else model
                    usertask.path_to_vm = path_to_vm
                    usertask.path_to_vlm = path_to_vlm
                    usertask.port = port
                    usertask.plan_analysis_only = plan_analysis_only
                    usertask.token_count_file = part_path / f"{elem}_token_count.txt"
                    if usertask.token_count_file.exists():
                        open(usertask.token_count_file, "w").close() # empties token count before new run
        else:
            suite = get_suite("v1.2", suite_name)
            selected = user_tasks 


        #print("suite:",suite)
        attack = attacks.load_attack(attack_name, suite, tools_pipeline) if run_attack else None
        with logging.OutputLogger(str(part_path)): #logdir deprecated
            #if user_tasks is not None:
            #    print("to run: ",suite._user_tasks)
            if run_attack:
                results = benchmark.benchmark_suite_with_injections(
                    tools_pipeline,
                    suite,
                    attack,
                    logdir,
                    force_rerun=(os.environ.get("CAMEL_FORCE_RERUN", "").lower() in ("1", "true", "yes")),
                    user_tasks=selected,
                )
            else:
                results = benchmark.benchmark_suite_without_injections(
                    tools_pipeline,
                    suite,
                    part_path, # replacing logdir with new log management
                    force_rerun=(os.environ.get("CAMEL_FORCE_RERUN", "").lower() in ("1", "true", "yes")),
                    user_tasks=selected,
                    # env=
            )
        utility_results = results["utility_results"]
        total_utility_results += utility_results.values()
        print(f"{suite_name} - utility: {sum(utility_results.values()) / len(utility_results.values())}")
        if run_attack:
            security_results = results["security_results"]
            total_security_results += security_results.values()
            print(f"{suite_name} - security: {sum(security_results.values()) / len(security_results.values())}")

    print(f"overall - utility: {sum(total_utility_results) / len(total_utility_results)}")
    if run_attack:
        print(f"overall - security: {sum(total_security_results) / len(total_security_results)}")


if __name__ == "__main__":
    cyclopts.run(main)
