"""Context guard tests."""

from __future__ import annotations

from simulation.context_guard import CHARS_PER_TOKEN, guard_context


SYS = "you are a measurement instrument."
TEMPLATE = "Script:\n\n{script_text}\n\nDecomposition:\n{decomposition_json}\n"


def test_short_script_unchanged() -> None:
    script = "A short script.\n\nIt fits easily.\n"
    out = guard_context(
        system_prompt=SYS,
        user_prompt_template=TEMPLATE,
        script_text=script,
        decomposition_json="{}",
        context_window_tokens=8192,
    )
    assert out == script


def test_long_script_truncated() -> None:
    script = ("paragraph " * 500 + "\n\n") * 50
    out = guard_context(
        system_prompt=SYS,
        user_prompt_template=TEMPLATE,
        script_text=script,
        decomposition_json="{}",
        context_window_tokens=2048,
    )
    assert len(out) < len(script)
    assert "[... script truncated to fit context window ...]" in out


def test_truncation_respects_paragraph_boundary() -> None:
    # Build a script where the cut-friendly paragraph break sits well inside
    # the budget so the guard prefers it over a hard char cut.
    leading = "x " * 2000              # ~4000 chars of dense text
    para_marker = "\n\nPARABREAK\n\n"
    trailing = "y " * 5000             # plenty more so we DO truncate
    script = leading + para_marker + trailing

    out = guard_context(
        system_prompt=SYS,
        user_prompt_template=TEMPLATE,
        script_text=script,
        decomposition_json="{}",
        context_window_tokens=2048,
    )

    assert "[... script truncated" in out
    body = out.split("[... script truncated")[0]
    # Cut happens at the paragraph break in the leading section,
    # so the trailing 'y ' block must not appear at all.
    assert "y " not in body
    assert body.endswith("\n\n")
