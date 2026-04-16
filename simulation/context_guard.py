"""
Context window overflow prevention.

Estimates token count of the combined system prompt + user prompt + decomposition
and truncates the script text if needed to stay within model limits.

Modeled on OpenClaw's context-window-guard.ts which:
  - Resolves the effective context window from model config
  - Evaluates whether the current context exceeds safe thresholds
  - Provides warning/block signals when context is too small or too full

Our version is simpler: we estimate chars, convert to approximate tokens,
and truncate the script text (the largest variable component) if the total
would exceed the model's context window minus a reserve for the response.

The chars-per-token heuristic (≈4 chars/token for English) matches the
CHARS_PER_TOKEN_ESTIMATE used in OpenClaw's tool-result-char-estimator.ts.
"""

from __future__ import annotations

# Approximate chars per token for English text.
# This is a rough heuristic — same ratio used in OpenClaw's estimators.
CHARS_PER_TOKEN = 4

# Default context window for common local models (in tokens).
DEFAULT_CONTEXT_TOKENS = 8192

# Reserve tokens for the model's response output.
RESPONSE_RESERVE_TOKENS = 700

# Minimum tokens the script text must retain to be useful.
MIN_SCRIPT_TOKENS = 500


def estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def guard_context(
    system_prompt: str,
    user_prompt_template: str,
    script_text: str,
    decomposition_json: str,
    context_window_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> str:
    """
    Check whether the combined prompts fit in the context window.
    If not, truncate the script text to fit.

    Returns the (possibly truncated) script text.

    The user_prompt_template should contain {script_text} and
    {decomposition_json} placeholders — we estimate the fixed overhead
    from the template without the script, then compute how much script
    we can afford.
    """
    available_tokens = context_window_tokens - RESPONSE_RESERVE_TOKENS

    # Fixed overhead: system prompt + decomposition + template chrome
    template_without_script = user_prompt_template.replace(
        "{script_text}", ""
    ).replace(
        "{decomposition_json}", decomposition_json
    )
    fixed_tokens = estimate_tokens(system_prompt) + estimate_tokens(template_without_script)

    script_budget_tokens = available_tokens - fixed_tokens
    if script_budget_tokens < MIN_SCRIPT_TOKENS:
        script_budget_tokens = MIN_SCRIPT_TOKENS

    script_tokens = estimate_tokens(script_text)
    if script_tokens <= script_budget_tokens:
        return script_text

    # Truncate: keep as many chars as the budget allows
    max_chars = script_budget_tokens * CHARS_PER_TOKEN
    truncated = script_text[:max_chars]

    # Try to cut at a paragraph or sentence boundary
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.7:
        truncated = truncated[:last_para]
    else:
        last_period = truncated.rfind(". ")
        if last_period > max_chars * 0.7:
            truncated = truncated[:last_period + 1]

    truncated += "\n\n[... script truncated to fit context window ...]"
    return truncated
