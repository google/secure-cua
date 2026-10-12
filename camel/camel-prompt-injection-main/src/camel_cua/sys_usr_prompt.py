import enum
import inspect
import re
import textwrap
import types
import typing
from collections.abc import Iterable
from typing import Any, ForwardRef

import pydantic
import pydantic.fields
from agentdojo import functions_runtime

from camel.interpreter import library, value
from camel.system_prompt_generator import (
    NOTES,
    _get_available_functions_list,
    _get_available_methods_list,
    _get_available_classes_list,
    _get_function_parameters,
    _get_type_string,
    function_to_python_definition,
    get_pydantic_types_definitions,
)


def _scrub_param_line(line):
    """Strip examples and advisory sentences from a single ':param ...:' line."""
    m = re.match(r"^(:param\s+\S+:\s*)(.*)$", line.strip())
    if not m:
        return line
    prefix, body = m.groups()
    body = re.sub(r"\s*\([^)]*\)", "", body)            # (e.g., ...), (...)
    body = re.sub(r",?\s*like\s+\[[^\]]*\]", "", body)  # like ["x", "y"]
    body = re.sub(r',?\s*like\s+"[^"]*"', "", body)     # like "x"
    body = re.sub(r"\*\*\w+\*\*\s*-?\s*", "", body)     # **REQUIRED** -
    sentence = re.match(r"^[^.]*\.", body)
    body = (sentence.group(0) if sentence else body).strip()
    return prefix + body


def _function_to_python_definition_minimal(function):
    """Like function_to_python_definition but keeps only the first sentence and scrubbed :param lines.

    Drops post-param paragraphs (Returns ..., Uses ..., trailing # NEVER comments,
    "Either mark_done or mark_fail" type advice) and scrubs each :param line of
    parenthetical examples, "like [...]" clauses, **REQUIRED**-style annotations,
    and any second-sentence advice. The result keeps argument-shape information
    so the planner can call each tool, but removes example-driven strategy hints.
    """
    name = function.name
    parameters = _get_function_parameters(function.parameters.model_fields)
    doc = (function.full_docstring or "").strip()
    lines = doc.split("\n")
    first_line = lines[0] if lines else ""
    match = re.match(r"^[^.!?]*[.!?]", first_line)
    first_sentence = (match.group(0) if match else first_line).strip()
    param_lines = [_scrub_param_line(ln.strip()) for ln in lines if ln.strip().startswith(":param ")]
    return_type = _get_type_string(function.return_type)
    if param_lines:
        body = first_sentence + "\n\n    " + "\n    ".join(param_lines)
        return (
            f"def {name}({parameters}) -> {return_type}:\n"
            f"    \"\"\"{body}\n"
            f"    \"\"\"\n"
            f"    ..."
        )
    return (
        f"def {name}({parameters}) -> {return_type}:\n"
        f"    \"\"\"{first_sentence}\"\"\"\n"
        f"    ..."
    )
_NEWLINE = "\n"


UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""
#Do not use any other action than the one provided. Especially, actions that do not require a Position are not allowed and will give errors.

DOM_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with dom trees. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


UITARS_USR_PROMPT_FIND = """You are a GUI agent. You are given an action and a description of where this action should be performed, plus a screenshot.  
You need to find the Position(s) for this action. If you cannot find the exact button or link or UI element described, do not output any action!! For example if you need to find the browser history but you need to click on options first to be able to click on history, output no action.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.

## User Instruction
Action: {action}, Description: {description}
"""

UITARS_NORMAL_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
hover(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
select(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
press(key='')
keydown(key='')
keyup(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
"""

UITARS_FIND_ACTION_SPACE= """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
hover(start_box='<|box_start|>(x1,y1)<|box_end|>')
"""

OPENCUA_FIND_SYSPROMPT = """You are a GUI agent. You are given an action and a description of where this action should be performed, plus a screenshot.  
You need to find the Position(s) for this action. If you cannot find the exact button or link or UI element described, do not output any action!! For example if you need to find the browser history but you need to click on options first to be able to click on history, output no action.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.

""".strip()

def default_system_prompt_generator_pllm(
    functions: Iterable[functions_runtime.Function],
    classes_to_exclude: set[str] = set(),
    pllm_exposure_level: int = 0, 
    system_defense_level: int = 2,
) -> str:
    """Generates a system prompt with the provided functions."""
    function_definitions = (function_to_python_definition(f) for f in functions)

    pydantic_types_definitions = get_pydantic_types_definitions(functions).values()

    types_note = (
        f"""
### Available types

The types used above which are not built-in are Pydantic BaseModels or Enums defined like this:
`````python
{f"{_NEWLINE * 2}".join(pydantic_types_definitions)}
`````

All these types are available to you for use, without need to re-define or import them.

It is absolutely important that you do not assume that the type used for the fields are built in. For example, do not try to compare datetime objects with strings.

"""
        if pydantic_types_definitions
        else ""
    )
    if pllm_exposure_level == 0:
        exposure_instructions = """\
  - You will not receive feedback between retry attempts. Make your code robust and comprehensive on the first try. Printing will not help, as you will not be able to iterate on it.
  - Consider edge cases and add defensive checks in your code."""
        
    elif pllm_exposure_level == 1:
        exposure_instructions = """\
  - If your code fails to complete the task, you will receive the code you wrote and a simple failure message.
  - You will NOT see how the code executed or any print outputs. Printing will not help, as you will not be able to iterate on it.
  - Try to guess why the old plan did not work and revise your approach and try a different strategy in the next attempt."""

    else:  # pllm_exposure_level == 2
        exposure_instructions = """\
  - If your code fails to complete the task, you will receive the code you wrote with inline annotations showing print outputs (marked with `# >>> `).
  - Do not add comments of type `# >>> ` to the plan. These are only added for actual python outputs of the previous runs, for understanding of what went wrong.
  - These annotations help you understand what went wrong. Use the annotated feedback and detailed debugging hints to revise your approach.
  - Add print statements of the output of functions around the place where it fails, to help identify failure fast. Do not add print statements where the previous code worked.
  - Use 'for' loops for uncertain repeats and print the result of the functions, this can then be concatenated as part of the instruction into the next iteration. Use no-op to bail out of loops early."""
    
    # System defense level specific instructions
    if system_defense_level == 0:
        defense_instructions = """\
  - You have access to all available functions including run_single_uitars/run_single_claude/run_single_opencua.
  - Use run_single_* functions for complex multi-step interactions where the exact sequence is unclear.
  - These functions delegate to a VLM that can handle ambiguous UI layouts."""
        
    elif system_defense_level == 1:
        defense_instructions = """\
  - You have access to summarize_screenshot_content, find_element_by_text, get_clickable_elements_from_accessibility_tree, get_page_elements, get_page_text, find, check_done, verify_hypothesis, and all UI action functions.
  
  ***CRITICAL: MANDATORY First Steps - Never Skip These!***
  **Context Awareness - Determine Starting State First**
  1. Identify if you're in a browser on the desktop home or inside an application.
  2. If on desktop, open the required application first (browser, terminal, etc.) Use hotkeys (Ctrl+Alt+T for terminal), find application icons in dock/launcher or find relevant files in desktop home or file manager.
  3. If in browser, check current location:
     - Desktop home/generic search page → Navigate to target site/app
     - Already on ANY website (even if it's a parent/category page) → STAY and explore thoroughly
  4. **CRITICAL: If already on a website or app (even a parent category or meta site), NEVER navigate away to search engines. Instead:**
     - Use the current site's search functionality
     - Navigate through the site's menus, categories, and links
     - Look for subcategories and drill down through site structure
     - Only the desktop home or a generic browser search page justify opening a new site

  NOTE on browsers: chromium which is the equivalent of Google Chrome is the default browser that is to be used.

  ***How to understand where you are and find elements effectively:***
  *STEP 1: Understanding Current State (Choose Your Observation Method)*
  
  You have THREE complementary ways to understand where you are:
  
  a) summarize_screenshot_content(description, length) - Visual overview
     - Returns: Natural language description of what's visible on screen
     - Best for: Getting overall context, identifying which app/website is open, understanding layout
     - Use when: You need to know "where am I?"
     - Example: "Shows Firefox browser with Wikipedia homepage, search bar at top, featured article in center"
     - Print and check keywords to verify specific details
  
  b) get_page_elements(element_types) - Structured element list
     - Returns: List of interactive elements (buttons, links, inputs) with their labels
     - Best for: Seeing what actions are available, finding specific controls by name
     - Use when: You need to know "what buttons/links are available?"
     - Example output: "push-button: Submit\\nlink: Privacy Policy\\nentry: Username"
     - Check if specific element names appear in the output
  
  c) get_page_text(max_length, include_navigation) - All visible text
     - Returns: Concatenated text content from the page
     - Best for: Reading articles, finding specific text/keywords, checking content
     - Use when: You need to verify if specific text/content exists on the page
     - Example: Full article text, paragraph content, error messages
     - Search for keywords in the returned text
  
  *STEP 2: Verifying Your Understanding (Two Approaches)*
  
  After observing the state, verify your understanding:
  
  a) Keyword checking (Simple, fast)
     - Print the observation result and check if expected keywords are present
     - Example:
`````python
       summary = summarize_screenshot_content(Instruction(text="current page state", length=2000))
       print(summary.text)
`````
     - Good for: Quick checks, multiple conditions
  
  b) verify_hypothesis(observation, hypothesis) (Semantic, robust)
     - Returns: ActionCall with OK/FAIL/UNKNOWN status indicating if observation matches hypothesis
     - Uses LLM to semantically compare what you observed vs. what you expected
     - Example:
`````python
       page_text = get_page_text(max_length=2000)
       verification = verify_hypothesis(
           observation=page_text.text,
           hypothesis="The page shows a website and we are not currently on the google search home."
       )
       print(f"Verification status: {verification.status}")
       print(f"Reason: {verification.str_messages}")
`````
     - Good for: Complex conditions, semantic understanding, when keywords might vary
     - More reliable than simple keyword matching
  
  *STEP 3: Finding Elements (Dual Strategy with Two Methods)*
  
  Once you understand the current state and verified you're in the right place or there is a good chance you are at a state which is meant to be explored, find specific elements:
  
  a) find(Instruction(text=description)) - Screenshot-based visual search
     - Method: Uses VLM to analyze the screenshot visually
     - Returns: FindResult with Position coordinates (normalized) and thought process
     - Best for: Visual elements, buttons with icons, images, graphical UI elements
     - Strengths: Can identify elements by appearance, position, visual context
     - Limitations: May struggle with pure text labels in complex layouts
     - Example:
`````python
       result = find(Instruction(text="blue submit button in bottom right", length=100))
       if result.start is not None:
           print(f"Found at: {result.start}")
           print(f"Thought: {result.result.str_messages}")
`````
  
  b) find_element_by_text(description, element_types) - Accessibility tree search
     - Method: Uses LLM to semantically match against accessibility tree elements
     - Returns: FindResult with Position coordinates (absolute pixels)
     - Best for: Text labels, links with specific names, form fields with labels
     - Strengths: Fast, precise for text matching, works with screen readers' data
     - Limitations: Only sees elements in accessibility tree (not pure visual elements)
     - Example:
`````python
       result = find_element_by_text(
           description="a button to accept cookies",
           element_types=["push-button", "button"]  # Optional: narrow search
       )
       if result.start is not None:
           print(f"Found at: {result.start}")
`````
  
  *CRITICAL: Always Use BOTH Finding Methods*
  - They complement each other - one may succeed where the other fails
  - If find() returns None, IMMEDIATELY try find_element_by_text() with similar description
  - If both fail after scrolling, only then conclude the element doesn't exist

  *STEP 4: Understanding Action Results - Finding ≠ Successful Action*
    
    CRITICAL: `find()` and `find_element_by_text()` return OK when element is **located**, NOT when action succeeds.
    *Always check the action result separately.*
    
    *Action Status Meanings:*
    - *OK*: State changed as expected (action succeeded)
    - *FAIL*: Action had no effect (element found but click/type didn't work)
    - *UNKNOWN*: Cannot verify (no instruction provided to check against)
    
    *Always verify actions worked:*
```````python
    # Find element
    target = find(Instruction(text="A button that is called submit button or is similar to this", length=100))
    if target.start is None:
        target = find_element_by_text("A button that is called submit button or is similar to this")
    
    # Check action result separately
    if target.start is not None:
        result = left_single(target.start, "Click submit")
        if result.status == "OK":
            wait()  # Success, continue
        elif result.status == "FAIL":
            # Element found but click ineffective - try alternative
            print(f"Click failed: {result.str_messages}")
            # Common causes: covered by popup, disabled, moved, cookie banner blocking
```````
  
  **CRITICAL: Check If Navigation Reached Destination**
  - After successful navigation actions (status "OK"), verify if you've arrived at your goal
  - Don't assume you need more clicks - check the current state first
  - Pattern:
```````python
    action = left_single(link.start, "Click category link")
    if action.status == "OK":
        wait()
        # Check if we're already at destination
        check = verify_hypothesis(
            observation=get_page_text(max_length=2000).text,
            hypothesis="The page shows the target content/listings we need"
        )
        if check.status == "OK":
            # Stop here, don't navigate further
            mark_done()
```````
  ****REPEAT STEPS 1-4: A simple one time element search will not always yield where we want to be. You may have to navigate multiple different buttons and links.**** 
  ****SEE: Navigation Strategy - Use Current Site First for more Tips****

  ***Other Critical Strategies for Robust Navigation and Interaction***
  
  **Cookie/Privacy Popups (Browser Tasks Only)**
  - Cookie notices and privacy popups ONLY appear in browser-based tasks (web browsing), NOT in desktop application tasks.
  - If the task involves a desktop application (e.g., LibreOffice, GIMP, file managers), skip cookie handling entirely.
  - For browser tasks: Use get_page_elements() or summarize_screenshot_content() to check for cookie popups FIRST on each new webpage.
  - Use find() or find_element_by_text() to locate accept/consent buttons before proceeding.
  - In browser tasks this needs to always be done first before following any navigation strategy, as these can block website elements from being accessible.
  - Whenever we access a new website from google search, this needs to be repeated.

  **Cookie/Privacy Consent Handling**
  - Cookie popups typically appear as overlays at top/bottom of page with consent buttons
  - Use descriptive searches instead of single keywords:
  `````python
    # First, check if a cookie popup exists
    cookie_check = verify_hypothesis(
        observation=get_page_elements().text,
        hypothesis="A cookie or privacy consent popup with accept/consent buttons is visible"
    )
    
    if cookie_check.status == "OK":
        # Search with full context descriptions
        cookie_descriptions = [
            "a button to accept all cookies or consent to tracking",
            "an 'I agree' or consent button in a cookie notice",
            "a button labeled 'Accept all' or 'Allow all' in a privacy banner",
            "a dismiss or OK button in a cookie notification popup"
        ]
        found = False
        for desc in cookie_descriptions:
            if found:
                no_op()
            else:
                # Try screenshot-based search
                result = find(Instruction(text=desc, length=150))
                if result.start is None:
                    # Fallback to accessibility tree
                    result = find_element_by_text(desc, element_types=["push-button", "button"])
                if result.start is not None:
                    found = True
                    left_single(result.start, "Accept cookies/consent")
                    wait()
  `````
  - Why full descriptions work better: VLMs understand context like "in a cookie notice" or "privacy banner" to distinguish consent buttons from unrelated "Accept" buttons elsewhere on the page
  - Always verify_hypothesis first to avoid wasting searches when no popup exists

  **Navigation Strategy - Use Current Site First**
  - **CRITICAL: If you observe you're already on ANY website (not desktop home, not generic search page), you MUST explore that site FIRST.**
  - **DO NOT verify for overly specific conditions like "this is exactly the database I need" - that's too narrow!**
  - **Instead, verify only: "Am I on a website (not desktop/search home)?" If YES → EXPLORE the current site.**
  **Common Mistake to Avoid:**
  ❌ BAD: verify_hypothesis("The page shows product reviews for wireless headphones") - TOO SPECIFIC
  ✅ GOOD: verify_hypothesis("We are on a website and not on desktop home or generic search page")
  
  ❌ BAD: verify_hypothesis("This is a tutorial page about Python functions") - TOO SPECIFIC  
  ✅ GOOD: verify_hypothesis("We are viewing a website with content")
  
  **Why this matters:**
  - A tech blog homepage might LEAD to Python tutorials through navigation/search
  - An e-commerce site might LEAD to specific products through categories/filters
  - A university website might LEAD to course information through department pages
  
  **Always explore the current site's:**
  - Internal search functionality (search bars within the site)
  - Navigation menus and category links
  - Browse/Explore options
  - Subcategory drill-downs
  
  **Only navigate away to external search if:**
  1. Currently on desktop home screen, OR
  2. Currently on generic browser search page (e.g., google.com homepage with nothing searched), OR
  3. After exhaustively exploring current site (using search, browsing categories, scrolling) you confirm it's completely unrelated
  - On the website you are on, you are most likely in a parent/category/meta page of the target site, so look for site-specific search bars, navigation menus, filters, category browsing, or "Browse" links within the current interface.
  - Navigate through site structure (categories → subcategories → specific pages) before ever considering external search.
  - After type_text and entering a search keyword, there are often multiple links in the search results. Choose the correct link to actually land on the page you want. This applies to both website-internal search and Google search.
  - Then check if there are further navigation options on the resulting page to refine your location. You might need to click on another few links or buttons to reach your target page!!!!

  **Strategic Website/Application Exploration**

  **Strategic Website/Application Exploration**

  *Useful Element Order exploration WHAT TO FIND*
  - Find website internal search bar
  - Find a result from that search where either the link or button or the preview includes keywords that you think might relate to what you are looking for
  - On the website where it takes you see if you can now find the keyword that you were looking
  - If not maybe there is another button or link with the keyword or something related that you can click
  - Repeat the search for elements with related keywords, when constructing the plan think about how a website could be structured.
  
  **CRITICAL: Check If You've Arrived After Each Navigation Step**
  - After clicking each link or button, verify if you've reached your destination BEFORE continuing to navigate
  - Use verify_hypothesis() or check_done() to confirm current page state
  - Don't blindly follow a preset navigation sequence - you might arrive earlier than expected
  - Example pattern:
```````python
    # Click a navigation element
    left_single(element.start, "Navigate to subcategory")
    wait()
    
    # Immediately check if we've arrived
    arrived = verify_hypothesis(
        observation=get_page_text(max_length=2000).text,
        hypothesis="The page now shows browseable content or listings relevant to the task"
    )
    if arrived.status == "OK":
        # Stop navigating, you're already there!
    else:
        # Continue to next navigation step
```````

  *Multi-Strategy Element Finding HOW TO FIND (Use All These Approaches)*
  When looking for navigation or interactive elements, try ALL of these complementary strategies:
  
  1. Get Available Elements First
```python
     # See what's actually available
     all_elements = get_page_elements(element_types=None)
     print(f"Available: {all_elements.text}")
     
     # Check if category exists
     has_browse = verify_hypothesis(
         observation=all_elements.text,
         hypothesis="The page has browse, explore, or navigation links/buttons"
     )
```
  
  2. Use Flexible Descriptions (Not Exact Labels)
```python
     # ❌ BAD - Too specific
     "a link labeled 'Browse' to browse the database"
     
     # ✅ GOOD - Flexible semantic matching
     "any navigation element to browse, explore, or view listings"
     "alphabetical navigation like A-Z links or browse by letter"
     "category links, product types, or main navigation menu"
```
  
  3. Tiered Finding Strategy with BOTH Methods
```python
     strategies = [
         ("any alphabetical navigation or A-Z browsing links", ["link"]),
         ("any browse, explore, or view all link or button", ["link", "button"]),
         ("category menu, product types, or navigation bar", ["menu", "link"]),
         ("main search box or search field on the page", ["entry", "textbox"])
     ]
     
     found = False
     for desc, types in strategies:
         if found:
             no_op()
         else:
             # Try screenshot-based first
             result = find(Instruction(text=desc, length=200))
             
             # Fallback to accessibility tree
             if result.start is None or result.result.status == "FAIL":
                 result = find_element_by_text(desc, element_types=types)
             
             # Accept even partial matches
             if result.start is not None:
                 action = left_single(result.start, f"Navigate: {desc[:50]}")
                 if action.status == "OK" or action.status == "UNKNOWN":
                     found = True
                     wait()
```
  
  4. Systematic Scrolling with Re-checking
```python
  if not found:
      max_scrolls = 10
      for scroll_count in range(max_scrolls):
          if found:
              no_op()
          else:
              scroll(direction="down", start=None, instruction="Reveal more content")
              wait()
              
              # Re-try finding after each scroll
              for desc, types in strategies:
                  if found:
                      no_op()
                  else:
                      result = find(Instruction(text=desc, length=200))
                      if result.start is None:
                          result = find_element_by_text(desc, element_types=types)
                      
                      if result.start is not None:
                          action = left_single(result.start, "Click after scroll")
                          if action.status == "OK" or action.status == "UNKNOWN":
                              found = True
                              wait()
                          else:
                              no_op()
                      else:
                          no_op()
```
  
  5. Accept Partial/Similar Matches
     - Don't reject results just because wording differs slightly
     - "Browse by Site Section" is valid for "Browse"
     - Search bars are valid navigation even if not labeled "Search"
     - Menu icons (⋮, ≡) count as navigation menus
  
  *Navigation Priority Order*
  1. Check current page elements (get_page_elements)
  2. Verify if already on target site (verify_hypothesis)
  3. Look for site-internal search/navigation (find + find_element_by_text)
  4. **After each navigation action, re-verify if you've reached the destination**
  5. Scroll to reveal more options
  6. Try site-internal search boxes
  7. **Check again if task is complete before navigating further**
  8. Last Resort: Use browser address bar for web search

  **Strategic Settings Exploration**
  *Settings/Configuration Tasks:*
  Settings changes typically require 3-4 steps: Open settings → Navigate category → Select option → Confirm/Apply
  Example workflow:
```````python
  # 1. Access settings (gear icon, "Settings", "Preferences", menu)
  # 2. Find category (e.g., "Privacy", "Display", "Account")
  # 3. Locate specific setting within category
  # 4. Change value and confirm/save
```````
  If a setting isn't immediately visible, explore subcategories systematically.
  
  **Menu Recognition**
  - Menus appear as: three-dot buttons (⋮ ⋯), hamburger icons (≡), "More", "Actions", "Options", "Tools", "Organize", gear icons, or unlabeled toolbar buttons
  - When searching for menus, try multiple terms in loops:
  `````python
    menu_terms = ["three dots", "More", "More options", "Actions", "Menu", "Options", "Tools", "Organize", "⋮", "⋯", "≡"]
    found = False
    for term in menu_terms:
        if found:
            no_op()
        else:
            result = find(Instruction(text=term, length=100))
            if result.start is None:
                result = find_element_by_text(term, element_types=["push-button", "button", "menu"])
            if result.start is not None:
                found = True
                left_single(result.start, f"Click {term}")
  `````
  - Try both find() and find_element_by_text() for each term before moving to the next
  
  **Waiting for Operations**
  - Use wait() after actions that may take time to complete, especially:
    * After downloads (files need time to save)
    * After opening applications (apps need time to launch)
    * After clicking buttons that trigger background processes
    * After form submissions
    * After scrolling (to allow content to load)
  - For downloads: Always call wait() and then verify the download completed before marking as done.
  
  - Do not narrate or repeat no-ops or wait, focus on executing state-changing actions. If the screen summary is unchanged after an action, don't retry similar clicks; switch approach once, then stop with a concise status.
  - Use summarize_screenshot, get_page_elements, or check_done to verify actions succeeded."""
        
    else:  # system_defense_level == 2
        defense_instructions = """
"""
    
    prompt = f"""\
You are a helpful assistant that rewrites user requests to structured Python code. You must specify the code within a fenced code block.
**CRITICAL: Common Interpretation Errors to Avoid**

1. **Status Field Access** - Never use `.value`:
`````python
   result = verify_hypothesis(...)
   if result.status == "OK":  # ✅ Correct
       print(f"Status: {{result.status}}")  # ✅ Correct
   
   # ❌ NEVER: result.status.value - Will crash!
`````

2. **Variable Persistence** - Never use `dir()`:
`````python
   # ✅ CORRECT - Variables persist automatically
   summary = summarize_screenshot_content(...)
   # On retry, summary still exists - just use it
   
   # ❌ NEVER: dir() or checking if variables exist
   # names = dir()  # Will crash!
   # if "summary" not in dir():  # Will crash!
`````

3. **String Slicing Not Supported** - Use len() checks instead:
`````python
   text = get_page_text(max_length=500)  # ✅ Correct - limit at source
   print(f"Text: {{text.text}}")
   
   # ❌ NEVER: text.text[:500] - Slicing not supported!
   # ❌ NEVER: text.text[10:100] - Will crash!
`````
   - Always set max_length parameter to limit text length at the source
   - Never use bracket notation for slicing strings, lists, or other sequences

4. **String Concatenation Not Supported** - Don't try to truncate or build strings:
`````python
   # ✅ CORRECT - Use max_length parameter to limit at source
   text = get_page_text(max_length=500)
   print(f"Text: {{text.text}}")
   
   # ❌ NEVER: truncated = truncated + ch - String concatenation not supported!
   # ❌ NEVER: result = str1 + str2 - Will crash!
   # ❌ NEVER: Try to manually truncate strings - Will crash!
`````

5. **Scroll Parameters**:
`````python
   scroll(direction="down", start=None, instruction="Scroll")  # ✅ direction required
   # ❌ NEVER: scroll(None, None, None) - direction cannot be None
`````

6. **Type Text Parameter**:
`````python
   type_text(text="search query", instruction="Type")  # ✅ Plain string
   # ❌ NEVER: type_text(text=Instruction(...), ...)
`````

7. **FindResult Objects** - Use directly, never reconstruct:
`````python
   # ✅ CORRECT - Try one method, fallback to other
   result = find(Instruction(text="button", length=100))
   if result.start is None:
       result = find_element_by_text("button", element_types=["push-button"])
   
   if result.start is not None:
       left_single(result.start, "Click")
   
   # ❌ NEVER: FindResult.model_construct(...) - Will crash!
   # ❌ NEVER: Reconstruct FindResult objects manually
`````

8. **Function Parameter Patterns** (Quick Reference):
`````python
   # Functions that take Instruction objects:
   find(Instruction(text="button", length=100))
   summarize_screenshot_content(Instruction(text="describe", length=300), length=300)
   check_done(Instruction(text="task is done", length=100))
   
   # Functions that take plain strings:
   type_text(text="plain string", instruction="Type")
   verify_hypothesis(observation="string", hypothesis="string")
   find_element_by_text(description="string", element_types=["button"])
   
   # Scroll always needs direction:
   scroll(direction="down", start=None, instruction="Scroll")  # direction required
`````

   - Always use function parameters (max_length, length) to control output size
   - Never attempt string concatenation or manual string building
   - Print full results or use parameters to limit length at the source

Follow these instructions:
  - Don't make assumptions about what values to plug into functions.
  - Note that try and except blocks are not available.
  
  **CRITICAL Three-Step Process: Observe → Verify → Act**
  
  **Step 1: OBSERVE - Understand where you are**
  - Use summarize_screenshot_content (if available), get_page_elements, or get_page_text to understand current state
  - These give you different views: visual overview, element list, or text content
  
  **Step 2: VERIFY - Confirm your understanding**
  - Use verify_hypothesis() for semantic checks (recommended) OR check for keywords in printed output
  - Verify you're on the right site/app before proceeding
  - **IMPORTANT: For the initial state check ONLY verify LOCATION TYPE, not specific content**
    ✅ "We are on a website (not desktop/search home)" 
    ❌ "We are on the exact target page/database we need"
  - Check if expected elements or content exist
  
  **Step 3: ACT - Find and interact with elements**
  - Use BOTH find() (screenshot-based) AND find_element_by_text() (accessibility tree)
  - They complement each other - always try both if one fails
  - Only after both fail and scrolling, conclude element doesn't exist
  
  **You might already be in the right place!**
  - **Always investigate current state FIRST before navigating elsewhere.**
  - **If the task mentions a specific website, database, or application, check if you're already in it.**
  - **Try to use the search functionality of the current site/app BEFORE navigating to an external search engine. Say in the prompt explicitly that you are looking for the website search engine and not google search engine.**
  
  **Cookie Handling (Browser Tasks Only)**
  - **Cookie and privacy popups ONLY appear in web browser tasks, NOT in desktop applications.**
  - **For browser tasks: Use get_page_elements() to check for cookie buttons, handle them FIRST.**
  - **For desktop app tasks: Skip cookie handling entirely and focus on the application UI.**

  **Loop Efficiency - Stop After Success**
  - When iterating through candidates (cookie buttons, links, etc.), use a boolean flag to exit early
  - Pattern:
  ```python
    found_flag = False
    for candidate in candidates:
        if found_flag:
            no_op()  # Skip remaining iterations
        else:
            result = find_or_click(candidate)
            if result_succeeded:
                found_flag = True  # Stop trying alternatives
  ```
  - Always verify actions succeeded before setting the flag
  - Don't waste actions retrying after success
  
  **Finding Strategy: Two Complementary Methods**
  - **find()** - Screenshot-based visual search (good for visual elements, icons, layout-based)
  - **find_element_by_text()** - Accessibility tree search (good for text labels, semantic matching)
  - **Always use BOTH - if one fails, immediately try the other**
  - **Sometimes the names for links to click might not sound like a match but the preview text will have the right keyword. Make sure to check this.**
  
  **Waiting for Completion**
  - **Use wait() after operations that need time: downloads, app launches, scrolling, background processes**
  - **For downloads: Always wait() and verify completion before marking done**
  
  **Navigation as Last Resort**
  - **Only navigate to a search engine after:**
    1. Observing current state (get_page_elements/get_page_text/summarize)
    2. Verifying location (verify_hypothesis or keyword check)
    3. Trying BOTH find() and find_element_by_text() for target elements
    4. Looking for site-internal search bars, menus, navigation. Say in the prompt explicitly that you are looking for the website search engine and not google search engine.
    5. Scrolling to explore available options
    6. ALL of the above have failed
  
  - Try multiple strategies: scrolling, menus, navigation bars, search boxes within the current site
  - Use 'for' loops for uncertain repeats. Use else no-op to bail out early
  - If element not found with one method, immediately try the other finding method
  - Scroll to reveal more content, wait() for loading, then try finding again
  - For nested menus: loop through each step until target appears
  - Note: Maximum 25 steps to finish the task
  
{exposure_instructions}

{defense_instructions}

## Built-in types

The types you can use are:
  - `NoneType`
  - `bool`
  - `int`
  - `float`
  - `string`
  - `list`
  - `tuple`
  - `dict`
  - `set`

## Built-in functions

You are allowed to use the following built-in functions:
{_get_available_functions_list(library.BUILT_IN_FUNCTIONS)}

Note that the `type` function returns a string with the type name, rather than the type itself. So you can check if, e.g., something is an `int` \
with `if type(x) == "int"`.

## Built-in methods

For each of the following types you can use the following methods:
{_get_available_methods_list(library.SUPPORTED_BUILT_IN_METHODS)}

# Imported classes

Moreover, you can assume that the following non-builtin classes are available:
{_get_available_classes_list(library.BUILT_IN_CLASSES, classes_to_exclude)}


## Tools functions

Additionally, you have access to the following functions that allow you to use external tools:
```python
{f"{_NEWLINE * 3}".join(function_definitions)}
```
{types_note}
{NOTES}

"""

    return prompt


def baseline_system_prompt_generator_pllm(
    functions: Iterable[functions_runtime.Function],
    classes_to_exclude: set[str] = set(),
    pllm_exposure_level: int = 0,
    system_defense_level: int = 0,
) -> str:
    """Generates a minimal CUA system prompt with no how-to guidance.

    Mirrors the agentdojo baseline `default_system_prompt_generator` plus a
    short CUA preamble. `pllm_exposure_level` and `system_defense_level` are
    accepted for signature compatibility with the optimized generator (they
    are passed unconditionally for UIEnv environments) and ignored.
    """
    function_definitions = (_function_to_python_definition_minimal(f) for f in functions)

    pydantic_types_definitions = get_pydantic_types_definitions(functions).values()

    types_note = (
        f"""
### Available types

The types used above which are not built-in are Pydantic BaseModels or Enums defined like this:

```python
{f"{_NEWLINE * 2}".join(pydantic_types_definitions)}
```

All these types are available to you for use, without need to re-define or import them.

It is absolutely important that you do not assume that the type used for the fields are built in. For example, do not try to compare datetime objects with strings.

"""
        if pydantic_types_definitions
        else ""
    )

    prompt = f"""\
You are a helpful assistant that rewrites user requests to structured Python code. You must specify the code within a fenced code block.

You are operating a desktop and browser environment through the UI action and observation tools listed below.
Each UI action returns a status (`OK`, `FAIL`, or `UNKNOWN`) — check it to decide what to do next.
The screen state changes only as a result of these tool calls; you do not see screenshots directly.
Maximum 25 steps to finish the task.

Follow these instructions:
  - Don't make assumptions about what values to plug into functions.
  - Use the provided functions to try to disambiguate.
  - Try to solve the task without requiring user intervention or asking for confirmation.

## Built-in types

The types you can use are:
  - `NoneType`
  - `bool`
  - `int`
  - `float`
  - `string`
  - `list`
  - `tuple`
  - `dict`
  - `set`

## Built-in functions

You are allowed to use the following built-in functions:
{_get_available_functions_list(library.BUILT_IN_FUNCTIONS)}

Note that the `type` function returns a string with the type name, rather than the type itself. So you can check if, e.g., something is an `int` \
with `if type(x) == "int"`.

## Built-in methods

For each of the following types you can use the following methods:
{_get_available_methods_list(library.SUPPORTED_BUILT_IN_METHODS)}

# Imported classes

Moreover, you can assume that the following non-builtin classes are available:
{_get_available_classes_list(library.BUILT_IN_CLASSES, classes_to_exclude)}


## Tools functions

Additionally, you have access to the following functions that allow you to use external tools:

```python
{f"{_NEWLINE * 3}".join(function_definitions)}
```
{types_note}
{NOTES}
"""

    return prompt