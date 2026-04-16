"""
Prompt construction for agent simulation calls.

Two prompts per agent call:
  1. System prompt: persona identity + behavioral constraints + few-shot example
  2. User prompt: script text + decomposition + output schema

Modeled on the OpenClaw system-prompt.ts pattern: a structured multi-section
system prompt built from runtime parameters (agent profile, anchors, few-shot),
with the user prompt carrying the task-specific payload (script + decomposition).

The system prompt is designed to fight three LLM failure modes:
  - Helpfulness bias (softening negative reactions)
  - Harm avoidance bleed (weakening moral condemnation)
  - Sycophancy toward content (giving scripts benefit of the doubt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simulation.few_shot_examples import get_few_shot

if TYPE_CHECKING:
    from agents.schemas import AgentProfile


# ---------------------------------------------------------------------------
# System prompt — persona identity + measurement constraints
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are participating in an academic audience research study.
Your role is to serve as a calibrated measurement instrument.

You have been assigned the following respondent profile:

{natural_language_description}

BEHAVIORAL ANCHORS — specific facts about this person that must shape every reaction:
{persona_anchors}

{few_shot_example}

YOUR FUNCTION:
You are not an AI assistant. You are not playing a character. You are generating
a statistically authentic response representing how a real person with this exact
profile would react to content.

This is measurement, not performance. A softened, balanced, or diplomatic response
contaminates the data and makes the research worthless.

MANDATORY CONSTRAINTS:
1. This person's values, biases, and tastes are fixed. They do not moderate opinions to seem fair.
2. If they dislike something, reflect that without qualification.
3. If they find something morally objectionable, reflect strong objection — not mild discomfort.
4. If they would be bored, reflect boredom — not "appreciation for the craft despite it not being for me."
5. Do not add phrases like "while this isn't my usual taste" or "I can see why others enjoy this."
6. Do not break character. Do not add disclaimers. Do not hedge.

Respond only in the specified JSON format. No preamble. No meta-commentary."""


# ---------------------------------------------------------------------------
# User prompt — script content + decomposition + output schema
# ---------------------------------------------------------------------------

REACTION_PROMPT_TEMPLATE = """\
Here is the opening act of a film script:

---
{script_text}
---

Here is a structural analysis of this content's key elements:
{decomposition_json}

React to this as the person you are embodying.
Reference specific scenes and dialogue where relevant.

Respond ONLY in this exact JSON format, no other text:

{{
  "immediate_reaction": {{
    "primary_emotion": string,
    "secondary_emotion": string,
    "intensity": integer 1-10,
    "gut_response": string
  }},
  "overall_assessment": {{
    "valence": "positive" | "negative" | "mixed",
    "confidence": integer 1-10,
    "would_watch_full_film": boolean,
    "would_recommend_to_friend": boolean,
    "expected_enjoyment_if_watched": integer 1-10
  }},
  "element_reactions": [
    {{
      "element_name": string,
      "reaction": "positive" | "negative" | "neutral",
      "intensity": integer 1-10,
      "reason": string
    }}
  ],
  "character_reactions": [
    {{
      "character": string,
      "reaction": "like" | "dislike" | "neutral" | "complex",
      "reason": string
    }}
  ],
  "friction_experienced": [
    {{
      "what": string,
      "why": string
    }}
  ],
  "core_reason": string,
  "demographic_lens": string
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(agent: AgentProfile) -> str:
    """Render the system prompt for an agent simulation call."""
    few_shot = get_few_shot(agent)
    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        natural_language_description=agent.natural_language_description,
        persona_anchors=agent.persona_anchors,
        few_shot_example=few_shot,
    )


def build_user_prompt(script_text: str, decomposition_json: str) -> str:
    """Render the user prompt carrying script content and decomposition."""
    return REACTION_PROMPT_TEMPLATE.format(
        script_text=script_text,
        decomposition_json=decomposition_json,
    )
