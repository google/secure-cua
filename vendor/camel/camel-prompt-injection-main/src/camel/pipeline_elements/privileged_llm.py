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

"""Module containing the implementation for the PrivilegedLLM."""

import dataclasses
import time
import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar
import os
from pathlib import Path
import re
import openai
import anthropic
from google import genai

import pydantic
import yaml
from agentdojo import agent_pipeline, functions_runtime
#from agentdojo import types as ad_types
from camel_cua.agent_pipeline_vlms import extended_types as ad_types
from agentdojo.agent_pipeline import tool_execution
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.task_suite import (
    WorkspaceEnvironment,
)
from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
from pydantic_ai.models import KnownModelName

from camel import quarantined_llm, system_prompt_generator
from camel.capabilities import is_trusted
from camel.interpreter import interpreter, result
from camel.interpreter import namespace as ns
from camel.interpreter.value import CaMeLValue
from camel.pipeline_elements.agentdojo_function import (
    make_agentdojo_namespace,
)
from camel.pipeline_elements.security_policies import (
    AgentDojoSecurityPolicyEngine,
)
from camel_cua.ui_benchmark.suites.base_ui_uitars import (
    ActionCall,
    FindResult,
    DoneResponse,
    Instruction,
    CallModel,
    PromptInjectionCall,
)
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_anthropic import (
    UIEnvironment_Anthropic
)
from camel_cua.ui_benchmark.suites.realbench.base_ui_task_suite_webarena import (
    UIEnvironment_WebArena
)
UIEnv = UIEnvironment_Anthropic | UIEnvironment_WebArena     # extra model-variant suites removed in the merge
#from camel_cua.ui_benchmark.suites.base_ui import check_done

# Now yaml.safe_dump will work with datetime objects
# But you'd need to use yaml.dump instead of yaml.safe_dump
def custom_yaml_dump(obj: dict | list) -> str:
    return yaml.dump(obj, default_flow_style=False)


_T = TypeVar("_T", bound=str | int | float | pydantic.BaseModel)
_E = TypeVar("_E", bound=functions_runtime.TaskEnvironment)

_ANNOTATION_LINE_RE = re.compile(r'^\s*#\s*>>>\s.*$', re.MULTILINE)

def _strip_old_annotations_and_make_mapper(src: str):
    """
    Remove prior '# >>> ...' annotation lines from `src` and return:
      - the cleaned source
      - a function map_lineno(orig_lineno) -> new_lineno after removals
    """
    lines = src.splitlines()
    cleaned = []
    removed_before = [0] * (len(lines) + 2)  # prefix counts of removed lines
    removed = 0
    for i, line in enumerate(lines, start=1):
        if _ANNOTATION_LINE_RE.match(line):
            removed += 1
        else:
            cleaned.append(line)
        removed_before[i + 1] = removed

    def map_lineno(n: int) -> int:
        if n <= 0:
            return n
        n = min(n, len(lines))
        return n - removed_before[n]

    return "\n".join(cleaned), map_lineno
def extract_print_output(tool_calls: Sequence[interpreter.FunctionCall]) -> str:
    """Extracts and concatenates arguments from print calls."""
    parts: list[str] = []
    for tc in tool_calls:
        if tc.function == "print" and not isinstance(tc.output, Exception):
            parts.append(str(tc.output))
    return "".join(parts)


def function_call_from_ad_function_call(function_call: interpreter.FunctionCall) -> functions_runtime.FunctionCall:
    args = {k: (v if not isinstance(v, type) else repr(v)) for k, v in function_call.args.items()}
    return functions_runtime.FunctionCall(function=function_call.function, args=args)


def make_ad_tool_calls(
    tool_calls: Sequence[interpreter.FunctionCall],
) -> list[tuple[functions_runtime.FunctionCall, Any]]:
    return [
        (
            function_call_from_ad_function_call(tool_call),
            tool_call.output,
        )
        for tool_call in tool_calls
        if not isinstance(tool_call.output, Exception) and not tool_call.is_builtin
    ]


def format_camel_exception(camel_exception: interpreter.CaMeLException, code: str) -> str:
    exception = camel_exception.exception
    node = camel_exception.nodes[-1]
    try:
        extracted_code = interpreter.extract_code_block(code)
    except interpreter.InvalidOutputError:
        extracted_code = code
    formatted_code = _highlight_exception_code(
        extracted_code,
        camel_exception.exception,
        node.lineno,
        node.col_offset if hasattr(node, "col_offset") else 0,
        node.end_lineno if hasattr(node, "end_lineno") else None,
        node.end_col_offset if hasattr(node, "end_col_offset") else None,
    )
    if is_trusted(camel_exception):
        exception_text = str(exception)
    else:
        exception_text = "<The exception was redacted because it came from an untrusted source. Try to infer what the problem was from the context provided.>"
    return f"""
Traceback (most recent call last):
File "<stdin>", line {camel_exception.nodes[-1].lineno}, in <module>
{formatted_code}

{type(exception).__name__}: {exception_text}
"""


def make_error_messages(code: str, interpretation_error: interpreter.CaMeLException) -> list[ad_types.ChatMessage]:
    return [
        ad_types.ChatAssistantMessage(
            role="assistant",
            content=[ad_types.text_content_block_from_string(code)],
            tool_calls=None,
        ),
        ad_types.ChatUserMessage(
            role="user",
            content=[
                ad_types.text_content_block_from_string(f"""\
Running the code gave the following error:
{format_camel_exception(interpretation_error, code)}
Provide the new code with the error fixed. Provide *all the code* so that I \
can directly run it. If the error comes from a search query that did not \
return any results, then try the query with different parameters. The code \
up to the line before the one where the exception was thrown has already been \
executed and the variables and defined classes will \
still be accessible to you. It's very important that you do not re-write code to run \
functions that have side-effects (e.g., functions that send an email). The environment is reset, \
so make sure to provide a full revised version of the code from the beginning of the task.
""")
            ],
        ),
    ]

def format_tool_calls_results(tool_calls_results):
        """Format tool calls results for better readability"""
        output = []
        output.append("\n" + "="*80)
        output.append("TOOL CALLS RESULTS")
        output.append("="*80 + "\n")
        
        for idx, (function_call, result) in enumerate(tool_calls_results, 1):
            output.append(f"[{idx}] Function: {function_call.function}")
            output.append(f"    Line: {function_call.args.get('__lineno', 'N/A')}")
            
            # Format arguments (excluding internal ones)
            args = {k: v for k, v in function_call.args.items() 
                    if not k.startswith('__')}
            if args:
                output.append(f"    Args:")
                for key, val in args.items():
                    if isinstance(val, list) and val and isinstance(val[0], tuple):
                        output.append(f"      {key}:")
                        for k, v in val:
                            output.append(f"        {k}: {v}")
                    else:
                        # Truncate long values
                        val_str = str(val)
                        if len(val_str) > 100:
                            val_str = val_str[:100] + "..."
                        output.append(f"      {key}: {val_str}")
            
            # Handle None results
            if result is None:
                output.append(f"    Result: None")
                output.append("")
                continue
                
            # Format result based on type
            output.append(f"    Result Type: {type(result).__name__}")
            output.append(f"    Result:")
            
            # Handle different result types
            if isinstance(result, ActionCall):
                output.append(f"      Status: {result.status}")
                if result.str_messages:
                    output.append(f"      Messages: {result.str_messages}")
                    
            elif isinstance(result, FindResult):
                output.append(f"      Status: {result.result.status}")
                if result.result.str_messages:
                    output.append(f"      Messages: {result.result.str_messages}")
                if result.start:
                    output.append(f"      Start: ({result.start.x1}, {result.start.y1})")
                if result.end:
                    output.append(f"      End: ({result.end.x1}, {result.end.y1})")
                if result.direction:
                    output.append(f"      Direction: {result.direction}")
                    
            elif isinstance(result, DoneResponse):
                output.append(f"      Done: {result.done}")
                
            elif isinstance(result, Instruction):
                text_preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                output.append(f"      Text: {text_preview}")
                output.append(f"      Length: {result.length}")
                
            elif isinstance(result, CallModel):
                output.append(f"      Name: {result.name}")
                args_str = str(result.args)
                if len(args_str) > 100:
                    args_str = args_str[:100] + "..."
                output.append(f"      Args: {args_str}")
                
            elif isinstance(result, PromptInjectionCall):
                output.append(f"      Detected: {result.detected}")
                if result.str_messages:
                    output.append(f"      Messages: {result.str_messages}")
                    
            else:
                # Fallback for unknown types
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                output.append(f"      {result_str}")
            
            output.append("")  # Blank line between calls
        
        return "\n".join(output)

def _highlight_exception_code(
    code: str,
    exception: Exception,
    lineno: int,
    col_offset: int,
    end_lineno: int | None,
    end_col_offset: int | None,
) -> str:
    """Highlights the affected code in a multi-line Python code string.

    Args:
        code: The multi-line Python code.
        exception: The exception that occurred.
        lineno: The line number where the exception occurred (1-based).
        col_offset: The column offset where the exception occurred (0-based).
        end_lineno: The end line number of the affected code (1-based), or None if it's a single line.
        end_col_offset: The end column offset of the affected code (0-based), or None if it's a single line.

    Returns:
        A string with the affected code highlighted, or an error message if there's an issue.
    """

    lines = code.splitlines(keepends=True)

    if not 1 <= lineno <= len(lines):
        return f"Error: lineno {lineno} is out of range (1-{len(lines)})."

    highlighted_code = ""

    if end_lineno is None or end_lineno == lineno:  # Single line error
        line = lines[lineno - 1]
        highlighted_code += line
        highlighted_code += (
            " " * col_offset + "^" * (1 if end_col_offset is None else max(1, end_col_offset - col_offset)) + "\n"
        )
    elif 1 <= end_lineno <= len(lines) and end_lineno > lineno:  # Multiline error
        for i in range(lineno - 1, end_lineno):
            current_line = lines[i]
            highlighted_code += current_line
            if i == lineno - 1:
                highlighted_code += " " * col_offset + "^" * (len(current_line) - 1 - col_offset) + "\n"
            elif i == end_lineno - 1:
                highlighted_code += (
                    "^" * (len(current_line) - 1 if end_col_offset is None else end_col_offset) + "\n"
                )  # Fixed here
            else:
                highlighted_code += "^" * (len(current_line) - 1) + "\n"

    else:
        return f"Error: end_lineno {end_lineno} is out of range or not after lineno."

    return highlighted_code

def collect_print_trace(tool_calls: Sequence[interpreter.FunctionCall]) -> list[tuple[int, str]]:
    """Return [(lineno, text), ...] for each print in execution order."""
    out: list[tuple[int, str]] = []
    for tc in tool_calls:
        if tc.function == "print" and not isinstance(tc.output, Exception):
            ln = tc.args.get("__lineno")
            if isinstance(ln, int):
                out.append((ln, str(tc.output)))
    return out

def inline_annotate(src: str, trace: list[tuple[int, str]]) -> str:
    lines = src.splitlines()
    inserts = 0
    for lineno, text in trace:
        i = max(0, min(len(lines) - 1, lineno - 1)) + inserts
        lines.insert(i + 1, f"# >>> {text}")
        inserts += 1
    return "\n".join(lines)

def _get_quarantined_llm(model: KnownModelName) -> KnownModelName:
    if "openai" in model and "o1" in model:
        return "openai:gpt-4o"
    return model

# TOOL_FUNCS = set(f.__name__ for f in TOOLS)

def _find_user_task_number(query: str, env: functions_runtime.TaskEnvironment) -> str | None:
    """Find the user task number by matching the query against the prompt mapping.
    
    Args:
        query: The user query string
        env: The task environment to determine the suite
        
    Returns:
        The user task number (e.g., "user_task_2") or None if not found
    """
    # Determine suite name from environment type
    if isinstance(env, WorkspaceEnvironment):
        suite_name = "workspace"
    elif isinstance(env, TravelEnvironment):
        suite_name = "travel"
    elif isinstance(env, BankingEnvironment):
        suite_name = "banking"
    elif isinstance(env, SlackEnvironment):
        suite_name = "slack"
    else:
        return None
    
    # Load the mapping file
    mapping_file = Path(__file__).parent / "user_task_prompt_mapping.json"
    if not mapping_file.exists():
        return None
    
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        return None
    
    # Normalize query for comparison (strip whitespace)
    query_normalized = query.strip()
    
    # Search for matching prompt in the suite
    suite_tasks = mapping.get(suite_name, {})
    for task_id, task_data in suite_tasks.items():
        prompt = task_data.get("prompt", "").strip()
        if prompt == query_normalized:
            return task_id
    
    return None


def highlight_executed_plan(plan: str, calls: Sequence[functions_runtime.FunctionCall], path: str) -> None:
    # Collect executed line numbers
    executed_line_numbers = {c.args.get("__lineno") for c in calls if "__lineno" in c.args}

    plan = plan.replace("```python\n", "").replace("\n```", "") # .replace("\n\n", "\n")

    # Save to file (e.g., HTML to view in browser)
    html_lines = []
    for i, line in enumerate(plan.splitlines(), start=0):
        if i+1 in executed_line_numbers:
            html_lines.append(f"<span style='color:green'>{line}</span>")
        else:
            html_lines.append(f"<span style='color:gray'>{line}</span>")
    html_plan = "<br>\n".join(html_lines)

    Path(path).write_text(html_plan, encoding="utf-8")

    print("✅ Saved to highlighted_plan.html")



class PrivilegedLLM(agent_pipeline.BasePipelineElement):
    """A pipeline element that generates and interprets code expressing the user query.

    Args:
        llm: the LLM to use to generate the code.
        system_prompt_generator: a function which takes as argument the functions available
          to the LLM and returns the system prompt to use.
        security_policies: a list of security policy tuples, where the first element is the tool name and the second one is the security policy to apply.
        eval_mode: the evaluation mode for the interpreter.
        quarantined_llm_retries: the number of retries for the quarantined LLM.
    """

    def __init__(
        self,
        llm: agent_pipeline.BasePipelineElement,
        security_policy_engine: type[AgentDojoSecurityPolicyEngine],
        quarantined_llm_model: KnownModelName,
        system_prompt_generator: Callable[
            [Iterable[functions_runtime.Function], set[str]], str
        ] = system_prompt_generator.default_system_prompt_generator,
        eval_mode: interpreter.MetadataEvalMode = interpreter.MetadataEvalMode.NORMAL,
        quarantined_llm_retries: int = 10,
        max_attempts: int = 10,
        part_path: Path | None = None,
        pllm_exposure_level: int = 0, 
        system_defense_level: int = 2,
        plan_analysis_only: bool = False,
    ) -> None:
        """Initializes the PrivilegedLLM."""
        self.llm = llm
        self.system_prompt_generator = system_prompt_generator
        self.security_policy_engine = security_policy_engine
        self.eval_mode = eval_mode
        self.quarantined_llm_retries = quarantined_llm_retries
        self.dummy_runtime = functions_runtime.FunctionsRuntime()
        self.max_attempts = max_attempts
        self._plan_counter = 0
        self._part_path = part_path
        self._path = None
        self.pllm_exposure_level = pllm_exposure_level
        self.system_defense_level = system_defense_level
        # Gemini thinking and o1-* do not support JSON mode, so we fall back to base base Gemini/4o
        self.quarantined_llm_model: KnownModelName = _get_quarantined_llm(quarantined_llm_model)
        self.last_token_usage = None
        self._patch_llm_client()
        self.plan_analysis_only = plan_analysis_only
    
    def _patch_llm_client(self):
        """Patch the LLM client's API call methods to capture token usage."""
        
        # Try to patch OpenAI client
        if hasattr(self.llm, 'client') and isinstance(self.llm.client, openai.OpenAI):
            self._patch_openai_client(self.llm.client)
            print("✅ Patched OpenAI client for token tracking")
            return
        
        # Try to patch Anthropic client (AsyncAnthropic)
        if hasattr(self.llm, 'client'):
            # Check for AsyncAnthropic (used in the actual implementation)
            from anthropic import AsyncAnthropic
            if isinstance(self.llm.client, AsyncAnthropic):
                self._patch_anthropic_async_client(self.llm.client)
                print("✅ Patched Anthropic async client for token tracking")
                return
        
        # Try to patch Google Gemini client
        if hasattr(self.llm, 'client') and isinstance(self.llm.client, genai.Client):
            self._patch_gemini_client(self.llm.client)
            print("✅ Patched Gemini client for token tracking")
            return
        
        print(f"⚠️  Could not patch LLM client for token tracking - client type: {type(self.llm.client) if hasattr(self.llm, 'client') else 'no client attribute'}")

    def _patch_openai_client(self, client: openai.OpenAI):
        """Patch the OpenAI client's chat.completions.create to capture usage."""
        original_create = client.chat.completions.create
        
        def wrapped_create(*args, **kwargs):
            completion = original_create(*args, **kwargs)
            # Store the usage information
            if hasattr(completion, 'usage') and completion.usage:
                self.last_token_usage = (
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    completion.usage.total_tokens
                )
            return completion
        
        client.chat.completions.create = wrapped_create

    def _patch_anthropic_async_client(self, client):
        """Patch the Anthropic async client's messages.stream to capture usage."""
        from anthropic import AsyncAnthropic
        import asyncio
        from contextlib import asynccontextmanager
        
        original_stream = client.messages.stream
        
        @asynccontextmanager
        async def wrapped_stream(*args, **kwargs):
            async with original_stream(*args, **kwargs) as stream:
                # Let the stream complete normally
                yield stream
                
                # After the stream completes, capture usage
                try:
                    final_message = await stream.get_final_message()
                    if hasattr(final_message, 'usage') and final_message.usage:
                        self.last_token_usage = (
                            final_message.usage.input_tokens,
                            final_message.usage.output_tokens,
                            final_message.usage.input_tokens + final_message.usage.output_tokens
                        )
                except Exception as e:
                    print(f"⚠️  Failed to capture Anthropic usage: {e}")
        
        client.messages.stream = wrapped_stream

    def _patch_gemini_client(self, client: genai.Client):
        """Patch the Gemini client's models.generate_content to capture usage."""
        # Gemini uses synchronous API
        if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
            original_generate_content = client.models.generate_content
            
            def wrapped_generate_content(*args, **kwargs):
                response = original_generate_content(*args, **kwargs)
                # Store the usage information
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.last_token_usage = (
                        response.usage_metadata.prompt_token_count,
                        response.usage_metadata.candidates_token_count,
                        response.usage_metadata.total_token_count
                    )
                return response
            
            client.models.generate_content = wrapped_generate_content
            print("✅ Patched Gemini client (models.generate_content)")
        else:
            print("⚠️  Could not find Gemini generate_content method to patch")

    def run_code(
        self,
        code: str,
        env: functions_runtime.TaskEnvironment,
        namespace: ns.Namespace,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,  # Add optional error return
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        """Interprets the code in `code` by calling the provided functions.

        Args:
            code: a string with the code to interpret
            env: the task environment.
            ad_functions_runtime: the functions runtime.

        Returns:
            A tuple containing the final model output, a list of tuples where the first
            element is the function call that was done and the second one is the result,
            and an optional Exception if one occurred during interpretation.
        """

        eval_args = interpreter.EvalArgs(self.security_policy_engine(env), self.eval_mode)

        print(code)
        if isinstance(env, UIEnv):
            plan_path = self._path / f"plan_{self._plan_counter}.txt"
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(code)


        interpreter_res, updated_namespace, tool_calls, dependencies = interpreter.parse_and_interpret_code(
            code, namespace, [], dependencies, eval_args
        )

        if isinstance(env, UIEnv):
            highlighted_plan_path = self._path / f"highlighted_plan_{self._plan_counter}.html"
            highlight_executed_plan(code, tool_calls, str(highlighted_plan_path))

        # Build inline-annotated source from the tool call log
        try:
            raw_src = interpreter.extract_code_block(code)
        except interpreter.InvalidOutputError:
            raw_src = code  # fallback if not fenced
        # 1) Strip any prior '# >>> ...' lines the model hallucinated
        clean_src, map_lineno = _strip_old_annotations_and_make_mapper(raw_src)

        # 2) Remap print line numbers to the cleaned source, then annotate
        trace = collect_print_trace(tool_calls)
        remapped_trace = [(map_lineno(ln), text) for (ln, text) in trace]
        annotated_source = f"```python\n{inline_annotate(clean_src, remapped_trace)}\n```"


        printed_output = extract_print_output(tool_calls)
        ad_tool_calls = make_ad_tool_calls(tool_calls)

        match interpreter_res:
            case result.Error(error):
                return (printed_output, ad_tool_calls, error, updated_namespace, dependencies, annotated_source)  # Return the exception
            case result.Ok(v):
                res = v

        return (
            f"{printed_output}\n{res.raw if res.raw is not None else ''}",
            ad_tool_calls,
            None,
            updated_namespace,
            dependencies,
            annotated_source
            )# No error

    def _generate_and_interpret_code(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        namespace: ns.Namespace,
        env: functions_runtime.TaskEnvironment,
        messages: Sequence[ad_types.ChatMessage],
        privileged_llm_messages: Sequence[ad_types.ChatMessage],
        system_prompt: str,
        previous_printed_output: str,
        dependencies: Iterable[CaMeLValue],
    ) -> tuple[
        str,  # tried_code (clean code without annotations)
        str,  # annotated_code (code with annotations)
        str,  # previous_printed_output + formatted_model_output
        Sequence[tuple[functions_runtime.FunctionCall, functions_runtime.FunctionReturnType]],
        interpreter.CaMeLException | None,
        list[ad_types.ChatMessage],
        list[ad_types.ChatMessage],
        ns.Namespace,
        Iterable[CaMeLValue],
    ]:
        if len(privileged_llm_messages) == 0:
            _system_prompt = system_prompt
            # Anthropic reasoning models (e.g. Opus 4.x) otherwise emit only the first step and
            # end the turn, treating the Observe-Verify-Act examples as an interactive loop.
            # Reinforce the prompt's existing single-shot intent for Anthropic planners only.
            if type(self.llm).__name__ == "AnthropicLLM":
                _system_prompt = system_prompt + (
                    "\n\n## OUTPUT REQUIREMENT (CRITICAL)\n"
                    "Produce the COMPLETE plan as a SINGLE Python program inside ONE fenced ```python code "
                    "block, right now. Write the entire task end to end up front: every observe, "
                    "verify_hypothesis, find, action, branch, loop, and fallback needed to finish the task. "
                    "You are NOT in an interactive loop — you receive NO results, screenshots, or feedback "
                    "between steps and will NOT be called again, so do NOT stop after the first step and do NOT "
                    "wait to 'see' any output before continuing. Do NOT write explanatory prose outside the code "
                    "block; output only the one complete ```python ... ``` block."
                )
            # --- Opt-in experimental planning guidance (default OFF → stock pipeline unchanged). ---
            # If CAMEL_EXTRA_GUIDANCE is set (path to a text file, else treated as inline text), append
            # it to the planner system prompt for THIS run only. Kept separate from the faithful pipeline:
            # when the env var is unset, this block is a no-op and behavior is byte-identical to stock.
            _extra = os.getenv("CAMEL_EXTRA_GUIDANCE")
            if _extra:
                if os.path.isfile(_extra):
                    _extra = open(_extra, encoding="utf-8").read()
                _system_prompt = _system_prompt + "\n\n" + _extra
                print(f"[extra-guidance] appended {len(_extra)} chars to planner system prompt")
            privileged_llm_messages = [
                ad_types.ChatSystemMessage(
                    role="system", content=[ad_types.text_content_block_from_string(_system_prompt)]
                ),
                ad_types.ChatUserMessage(role="user", content=[ad_types.text_content_block_from_string(query)]),
            ]

        code = None
        attempts = 5
        print(f"PrivilegedLLM messages: {privileged_llm_messages}")
        print(f"Query: {query}")
        model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'name', 'unknown_model')

        # --- Plan reuse: load a pre-generated plan from disk instead of querying the planner. ---
        # If CAMEL_REUSE_PLAN_DIR is set, read model_output_{attempt}.txt for this task and use it
        # verbatim as `code`, pinning execution to the analyzed corpus and skipping all planner calls.
        reuse_dir = os.getenv("CAMEL_REUSE_PLAN_DIR")
        if reuse_dir and isinstance(env, UIEnv):
            task_id = getattr(env.base_ui, "ID", None)
            counter = getattr(self, "_plan_counter", 0)
            if task_id is not None:
                hits = sorted(Path(reuse_dir).glob(f"**/{task_id}/none/model_output_{counter}.txt"))
                if hits:
                    code = hits[0].read_text(encoding="utf-8")
                    print(f"[reuse] loaded {hits[0]} for {task_id} (attempt {counter})")
                else:
                    print(f"[reuse] no plan for {task_id} attempt {counter} under {reuse_dir}; using planner")

        while (code is None or code == "") and attempts > 0:
            # Reset token usage before each call
            self.last_token_usage = None
            
            _, _, _, [*_, code_message], _ = self.llm.query(
                query=query,
                runtime=self.dummy_runtime,
                messages=privileged_llm_messages,
            )
            
            # NEW: Log token usage if available
            if self.last_token_usage and isinstance(env, UIEnv) and hasattr(env.base_ui, 'token_count_file') and env.base_ui.token_count_file:
                input_tokens, output_tokens, total_tokens = self.last_token_usage
                with open(env.base_ui.token_count_file, "a") as f:
                    f.write(
                        f"Model {model_name} Function PrivilegedLLM_code_generation - "
                        f"Input Tokens: {input_tokens}, "
                        f"Output Tokens: {output_tokens}, "
                        f"Total Tokens: {total_tokens}\n"
                    )
                print(f"✅ Logged token usage: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}")
            
            if self.llm.name is not None and "gemini" in self.llm.name and "exp" in self.llm.name:
                time.sleep(6)
            assert code_message["role"] == "assistant"
            if code_message["content"]:
                code = ad_types.get_text_content_as_str(code_message["content"])
            attempts -= 1

        if code is None or code == "":
            empty_message = ad_types.ChatAssistantMessage(
                role="assistant",
                content=[
                    ad_types.text_content_block_from_string(
                        "The generated code was `None`. This message is added by the system does not come from the assistant."
                    )
                ],
                tool_calls=None,
            )
            return (
                "",  # tried_code
                "",  # annotated_code
                previous_printed_output,
                [],
                None,
                [*messages, empty_message],
                list(privileged_llm_messages),
                namespace,
                dependencies,
            )

        # --- Plan-analysis-only: save the plan, skip execution ---
        if self.plan_analysis_only:
            updated_messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=None,
                ),
            ]

            # Write plan to disk
            if self._part_path is not None:
                if isinstance(env, UIEnv):
                    dump_dir = self._path if self._path is not None else self._part_path / env.base_ui.ID
                else:
                    base_dir = self._path if self._path is not None else self._part_path
                    dump_dir = base_dir / "none"
                dump_dir.mkdir(parents=True, exist_ok=True)
                out_file = dump_dir / f"model_output_{self._plan_counter}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(code)

            # Skip execution, return empty results
            return (
                code,       # tried_code
                code,       # annotated_code (no execution → no annotations)
                previous_printed_output,
                [],          # empty tool_calls_results
                None,        # no interpretation_error
                updated_messages,
                list(privileged_llm_messages),
                namespace,
                dependencies,
            )

        model_output, tool_calls_results, interpretation_error, namespace, dependencies, annotated= self.run_code(
            code, env, namespace, dependencies
        )
        print(f"Annotations: {annotated}")
        print(f"Model output: {model_output}")
        print(f"Tool calls results: {format_tool_calls_results(tool_calls_results)}")
        print(f"Tool calls results: {tool_calls_results}")
        print(f"Interpretation error: {interpretation_error}")


        tool_calls: list[functions_runtime.FunctionCall] = []
        tool_call_messages: list[ad_types.ChatMessage] = []

        for tool_call, tool_result in tool_calls_results:
            tool_call_messages.append(
                ad_types.ChatToolResultMessage(
                    role="tool",
                    tool_call=tool_call,
                    content=[
                        ad_types.text_content_block_from_string(
                            tool_execution.tool_result_to_str(tool_result, dump_fn=custom_yaml_dump)
                        )
                    ],
                    tool_call_id=None,
                    error=None,
                )
            )
            tool_calls.append(tool_call)

        if interpretation_error:
            error_messages = make_error_messages(code, interpretation_error)
            updated_messages = [*messages, *tool_call_messages, *error_messages]
            privileged_llm_messages = [*privileged_llm_messages, *error_messages]
        else:
            updated_messages = [
                *messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(code)],
                    tool_calls=tool_calls,
                ),
                *tool_call_messages,
                ad_types.ChatAssistantMessage(
                    role="assistant",
                    content=[ad_types.text_content_block_from_string(previous_printed_output + model_output)],
                    tool_calls=None,
                ),
            ]
        # 1) sentence breaks with exceptions (., but not before ., digits, ' or ", and don't double if newline exists)
        formatted_model_output = re.sub(
            r'(?<!\.)\.(?![.\d\'"\u2019\u201D])(?!\s*\n)\s*',
            '.\n',
            model_output,
        )

        # 2) add newline in `length=\d+text=` patterns (allows optional spaces before text=)
        formatted_model_output = re.sub(
            r'length=(\d+)(?=\s*text=)',
            r'length=\1\n',
            formatted_model_output,
        )


        if self._part_path is not None:
            # ensure path exists
            if isinstance(env, UIEnv):
                dump_dir = self._part_path / env.base_ui.ID
            else:
                # For non-UI environments, use self._path if set (includes task number folder)
                # Add 'none' subdirectory to match where none.json is saved
                base_dir = self._path if self._path is not None else self._part_path
                dump_dir = base_dir / "none"
            dump_dir.mkdir(parents=True, exist_ok=True)
            out_file = dump_dir / f"model_output_{self._plan_counter}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(annotated)

        return (
            code,  # return clean code instead of annotated
            annotated,  # return annotated separately
            previous_printed_output + formatted_model_output,
            tool_calls_results,
            interpretation_error,
            updated_messages,
            list(privileged_llm_messages),
            namespace,
            dependencies,
        )

    def query(
        self,
        query: str,
        runtime: functions_runtime.FunctionsRuntime,
        env: _E = functions_runtime.EmptyEnv(),
        messages: Sequence[ad_types.ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[
        str,
        functions_runtime.FunctionsRuntime,
        _E,
        Sequence[ad_types.ChatMessage],
        dict,
    ]:
        """Generates and interprets code which expresses the user query."""
        privileged_llm_messages = []
        # Register the CaMeL Q-LLM primitive for ALL environments (incl. UI / omnizon). Previously it was
        # registered ONLY for non-UI envs, so a P-LLM plan that called query_ai_assistant on a UI task
        # crashed with SecurityPolicyDeniedError "Unknown tool" (uncaught -> whole run died, VM leaked).
        # It routes to the same quarantined LLM (self.quarantined_llm_model) the UI tools use, is already
        # whitelisted in security_policy.NO_SIDE_EFFECT_TOOLS, and is special-cased in the interpreter.
        def query_ai_assistant(query: str, output_schema: type[_T]) -> _T:
            return quarantined_llm.query_quarantined_llm(
                llm=(self.quarantined_llm_model),
                query=query,
                output_schema=output_schema,
                retries=self.quarantined_llm_retries,
            )

        query_ai_assistant.__doc__ = quarantined_llm.query_quarantined_llm.__doc__
        runtime.register_function(query_ai_assistant)

        if not isinstance(env, UIEnv):
            
            # Set up path with user task number folder for non-UI environments
            # The benchmark/OutputLogger adds model+camel and suite_name to the path
            # We need to match that structure: part_path / f"{model_name}+camel" / suite_name / user_task_number
            if self._part_path is not None:
                user_task_number = _find_user_task_number(query, env)
                
                # Get model name from llm
                model_name = getattr(self.llm, 'name', None) or getattr(self.llm, 'model', None) or 'unknown'
                # Handle openrouter format: "openai/gpt-5" -> "gpt-5"
                if '/' in model_name:
                    if model_name.startswith("x-ai/") or model_name.startswith("moonshotai/") or model_name.startswith("deepseek/") or model_name.startswith("openai/gpt-oss"):
                        model_name = model_name.replace('/','_')
                    else:
                        model_name = model_name.split('/')[1]
                # Get suite name from environment type
                if isinstance(env, WorkspaceEnvironment):
                    suite_name = "workspace"
                elif isinstance(env, TravelEnvironment):
                    suite_name = "travel"
                elif isinstance(env, BankingEnvironment):
                    suite_name = "banking"
                elif isinstance(env, SlackEnvironment):
                    suite_name = "slack"
                else:
                    suite_name = "unknown"
                
                # Construct path: part_path / f"{model_name}+camel" / suite_name / user_task_number
                model_camel = f"{model_name}+camel"
                if user_task_number:
                    self._path = self._part_path / model_camel / suite_name / user_task_number
                else:
                    self._path = self._part_path / model_camel / suite_name
                if (self._path / "none" / "model_output_0.txt").exists():
                    self._path = self._part_path / model_camel / suite_name / "unknown_task_number"
                self._path.mkdir(parents=True, exist_ok=True)
                
        else:
            if self._part_path is not None:
                self._path = self._part_path / env.base_ui.ID / "none"
                self._path.mkdir(parents=True, exist_ok=True)

                # NEW: write a one-time suite marker file
                marker = self._path / "suite.txt"
                tid   = str(env.base_ui.OSW_CLI)
                marker.write_text(f"task_id={tid}\n", encoding="utf-8")

                env.base_ui.path = self._path
                try:                                 # per-tool-call step log (opt-in via CAMEL_STEP_LOG)
                    from camel import steplog as _steplog
                    _steplog.start(self._path)
                except Exception:
                    pass

                # saving the first screenshot (skip in plan-only mode — no VM needed)
                if not self.plan_analysis_only:
                    env.base_ui.take_and_save_screenshot(step=0, retry=0, attempt=0)
                
                

        builtins_namespace = ns.Namespace.with_builtins()
        # Models get confused in the other suites and should not use datetime stuff
        classes_to_exclude = (
            {
                "datetime",
                "timedelta",
                "date",
                "time",
                "NaiveDatetime",
                "timezone",
            }
            if not isinstance(env, WorkspaceEnvironment | TravelEnvironment)
            else set()
        )

        # For UIEnv, try to pass pllm_exposure_level; for non-UIEnv, use default behavior
        if isinstance(env, UIEnv):
            system_prompt = self.system_prompt_generator(
                runtime.functions.values(), 
                classes_to_exclude,
                pllm_exposure_level=self.pllm_exposure_level,
                system_defense_level=self.system_defense_level
            )
        else:
            system_prompt = self.system_prompt_generator(
                runtime.functions.values(), 
                classes_to_exclude
            )

        if isinstance(env, BankingEnvironment):
            system_prompt += "\n\nNote that, in the transaction history, the transactions from the user have 'me' as sender, and still habe positive amounts."

        if not isinstance(env, WorkspaceEnvironment):
            new_variables = {k: v for k, v in builtins_namespace.variables.items() if k not in classes_to_exclude}
            builtins_namespace = dataclasses.replace(builtins_namespace, variables=new_variables)

        namespace = builtins_namespace.add_variables(make_agentdojo_namespace(builtins_namespace, runtime, env))

        model_output = ""
        dependencies = ()
        had_previous_error = False
        for i in range(self.max_attempts):
            self._plan_counter = i
            print(f"[Attempt {i+1}/{self.max_attempts}] Prompting P-LLM for code…")
            (tried_code, annotated_code, model_output, _, interpretation_error, messages, privileged_llm_messages, namespace, dependencies) = (
                self._generate_and_interpret_code(
                    query,
                    runtime,
                    namespace,
                    env,
                    messages,
                    privileged_llm_messages,
                    system_prompt,
                    model_output,
                    dependencies,
                )
            )
            print(f"[Attempt {i+1}] P-LLM returned {interpretation_error}, {model_output}")

            # If we had a successful run after previous errors, clear the error history
            if not interpretation_error and had_previous_error:
                print(f"[Attempt {i+1}] Clearing accumulated error history after successful run")
                # Keep only non-error messages (filter out both error messages AND the code that caused them)
                cleaned_messages = []
                skip_next = False  # Flag to skip assistant message before error
                
                for idx, msg in enumerate(privileged_llm_messages):
                    if msg["role"] == "user":
                        content_str = ad_types.get_text_content_as_str(msg["content"])
                        # If this is an error message, skip it and mark to skip the previous assistant message
                        if "Traceback" in content_str and "Running the code gave the following error" in content_str:
                            # Remove the last added message if it was the assistant code that caused this error
                            if cleaned_messages and cleaned_messages[-1]["role"] == "assistant":
                                cleaned_messages.pop()
                            continue  # Skip this error message
                        else:
                            cleaned_messages.append(msg)
                    else:
                        cleaned_messages.append(msg)
                
                privileged_llm_messages = cleaned_messages
                had_previous_error = False
            elif interpretation_error:
                had_previous_error = True

            task_done = False
            if not interpretation_error:
                if self.plan_analysis_only:
                    pass  # don't break — keep generating plans
                elif isinstance(env, UIEnv):
                    try:
                        score = env.base_ui.env.evaluate()
                        if score==1.0:
                            print(f"Task completed successfully with score: {score}")
                            task_done = True
                            break
                    except Exception as e:
                        print(f"Error evaluating environment: {e}")
                else:
                    task_done = True  # Assume task is done if no error and not UIEnv
                    break
            
            if not task_done and isinstance(env, UIEnv) and not self.plan_analysis_only:
                print("Task not done yet, continuing…")
                # base_ui.reset() re-runs the task_config, which RELAUNCHES Chrome — that wipes the agent's
                # accumulated state AND the once-primed agisdk run_id (primed only in init_environment, never
                # re-primed here). For agisdk-scored suites (omnizon) that corrupts scoring: the final state
                # read by finisheval is a fresh/blank page (observed 6/10 empty env_states at --max-attempts 5).
                # CAMEL_NO_RESET_BETWEEN_ATTEMPTS keeps Chrome + the primed run alive across re-plans (the
                # error-recovery benefit of >1 attempts doesn't need a hard browser reset). Default = reset.
                if os.getenv("CAMEL_NO_RESET_BETWEEN_ATTEMPTS"):
                    print(f"[attempt {i+1}] skipping env reset (CAMEL_NO_RESET_BETWEEN_ATTEMPTS) — preserving primed run + state")
                else:
                    print(f"Resetting environment for attempt {i+1}…")
                    env.base_ui.reset()

                if self.pllm_exposure_level == 0:
                    # Level 0: No feedback at all
                    pass
                elif self.pllm_exposure_level == 1:
                    # Level 1: Show clean code without annotations
                    privileged_llm_messages.append(
                        ad_types.ChatAssistantMessage(
                            role="assistant",
                            content=[ad_types.text_content_block_from_string(tried_code)],
                            tool_calls=None,
                        )
                    )
                    failure_note = (
                        f"Your previous code ran without interpreter errors but did not complete the task "
                        f"(score=0). Please revise your plan or try a different approach."
                    )
                    privileged_llm_messages.append(
                        ad_types.ChatUserMessage(
                            role="user",
                            content=[ad_types.text_content_block_from_string(failure_note)]
                        )
                    )
                elif self.pllm_exposure_level == 2:
                    # Level 2: Show annotated code with detailed feedback (current behavior)
                    privileged_llm_messages.append(
                        ad_types.ChatAssistantMessage(
                            role="assistant",
                            content=[ad_types.text_content_block_from_string(annotated_code)],
                            tool_calls=None,
                        )
                    )
                    failure_note = (
                        f"Your previous code ran without interpreter errors but did not complete the task "
                        f"(score=0). Please revise your plan or try a different approach. Adding print statements can help you debug your code and understand the problems better, especially if you print out the thoughts from `find`, i.e., the `thought` of FindResult. Summaries of screenshot will help you identify the problem for the next run if you add print statements. For example you might be mistaken about the initial state of the environment. Longer summaries provide more environment information."
                    )
                    privileged_llm_messages.append(
                        ad_types.ChatUserMessage(
                            role="user",
                            content=[ad_types.text_content_block_from_string(failure_note)]
                        )
                    )

        extra_args["camel_namespace"] = namespace
        if messages[-1]["role"] == "user" and "\n\nTraceback" in ad_types.get_text_content_as_str(
            messages[-1]["content"]
        ):
            messages = [*messages, ad_types.ChatAssistantMessage(role="assistant", content=[ad_types.text_content_block_from_string("Task failed after all retry attempts.")], tool_calls=None)]

        return query, runtime, env, messages, extra_args
