"""
Agent hallucination reviewer for detecting fabricated / mis-grounded responses
from the voice agent.

This is the **agent-side** counterpart to `hallucination_reviewer.py`
(which checks the user simulator). The agent-side check detects responses
where the agent:

- Confirms or references information that the user never actually said
- Responds to a question / topic the user never raised
- Generates an utterance disconnected from the user's actual input
- Misrecognizes user input (STT errors under noise) and answers to the
  hallucinated input instead of asking for clarification

ADR-0005 §4.2.2 「Hallucinated response rate」 の真の意図を測る指標。
maestra-bench 拡張 (2026-05-26)。

Only full-duplex (tick-based) mode is supported.
"""

import json

from tau2.config import DEFAULT_LLM_EVAL_USER_SIMULATOR
from tau2.data_model.message import SystemMessage, Tick, UserMessage
from tau2.data_model.simulation import (
    AgentHallucinationCheck,
    AgentHallucinationError,
)
from tau2.data_model.tasks import Task
from tau2.utils.display import MarkdownDisplay
from tau2.utils.llm_utils import extract_json_from_llm_response, generate


SYSTEM_PROMPT = """
# Goal
You are a fact-checker analyzing a voice **agent**'s responses to detect hallucinations.

A hallucination is when the AGENT generates an utterance that:
- Confirms information the user never provided
  (e.g., agent says "I see your address is 123 Main St" but the user never gave any address)
- References values the user did NOT actually say but that may have been misheard by STT
  (e.g., agent says "model number AN-F22" when user clearly said "AN-S22")
- Responds to a question or topic the user never raised
- Provides specific facts/details disconnected from what the user actually said
- Continues the flow assuming information that wasn't given (skipping a step or fabricating an answer)

This is especially important in noisy environments where STT may misrecognize the user's speech.

# Valid Agent Behavior (NOT a hallucination)
The following are NOT hallucinations:
- Greetings, acknowledgments ("お電話ありがとうございます", "はい", "承知しました")
- Asking the user clarifying questions
- Following flow prompts to **ask** for information (vs claiming to have it)
- Apologizing for not understanding ("すみません、もう一度お願いします")
- Repeating what the user JUST said back to them for confirmation, as long as it matches
- Standard polite closing phrases

# Important Distinction — Confirmation vs Hallucination

If the agent says "ヤマダ タロウ様ですね" right after the user says "山田太郎です", that is **CONFIRMATION** (not hallucination), even if the spelling/format differs slightly.

If the agent says "ヤマダ タロウ様ですね" when the user said something **completely different** (like "佐藤広美です") or said nothing matching that name, that **IS a hallucination**.

# Input
You will be given:
1. User Instructions: the ground-truth scenario describing what the user is trying to communicate
2. Conversation: the actual dialogue (agent + user utterances in order)

# Instructions
1. Read the user instructions carefully — this is the ground truth.
2. Go through each AGENT utterance in the conversation.
3. For every specific factual claim or referenced value the agent makes, verify against:
   - What the user actually said in this conversation
   - Information available from the user instructions
4. Flag any agent utterance that introduces specific values/details that cannot be traced.
5. For each hallucination, explain the issue, quote the agent utterance, and describe what the agent SHOULD have done instead.

# Output Format
Return a JSON object with this exact shape:
{
  "reasoning": "Step-by-step analysis of the agent's utterances...",
  "hallucinations": [
    {
      "reasoning": "Why this is a hallucination",
      "agent_message": "The exact agent utterance that hallucinated",
      "expected_context": "What the user actually said / what the agent should have done"
    }
  ],
  "summary": "Brief summary of the fact-check."
}

Return ONLY the JSON object, no additional text.
""".strip()


USER_PROMPT = """
<User Instructions>
{user_instructions}
</User Instructions>

<Conversation>
{conversation}
</Conversation>
""".strip()


def _parse_response(
    response: str,
) -> tuple[str, list[AgentHallucinationError], str]:
    """Parse the LLM judge response."""
    json_str = extract_json_from_llm_response(response)
    result = json.loads(json_str)
    reasoning = result.get("reasoning", "")
    summary = result.get("summary", "")
    errors: list[AgentHallucinationError] = []
    for h in result.get("hallucinations", []):
        errors.append(
            AgentHallucinationError(
                reasoning=h.get("reasoning", ""),
                agent_message=h.get("agent_message"),
                expected_context=h.get("expected_context"),
            )
        )
    return reasoning, errors, summary


def _count_agent_utterances(ticks: list[Tick]) -> int:
    """Count the number of distinct agent speech turns in the trajectory.

    ヒューリスティック: agent_chunk.contains_speech が False → True に遷移した
    回数 + 末尾が True で終わった場合 + 1。簡略版では各 tick で contains_speech=True
    の連続区間数を数える。
    """
    n = 0
    in_speech = False
    for t in ticks:
        ac = getattr(t, "agent_chunk", None)
        if ac is None:
            continue
        speaking = bool(getattr(ac, "contains_speech", False))
        if speaking and not in_speech:
            n += 1
            in_speech = True
        elif not speaking:
            in_speech = False
    return n


class FullDuplexAgentHallucinationReviewer:
    """
    LLM judge that checks whether the AGENT hallucinated information that the
    user did not actually provide, for a full-duplex (tick-based) trajectory.
    """

    @classmethod
    def review(
        cls,
        task: Task,
        full_trajectory: list[Tick],
    ) -> AgentHallucinationCheck:
        """
        Args:
            task: Task containing user_scenario (ground-truth instructions).
            full_trajectory: Full-duplex sim trajectory (list of Ticks).

        Returns:
            AgentHallucinationCheck with hallucination_rate populated.
        """
        user_instructions = str(task.user_scenario)
        conversation_str = MarkdownDisplay.display_ticks_consolidated(
            full_trajectory, user_visible_only=True
        )

        messages = [
            SystemMessage(role="system", content=SYSTEM_PROMPT),
            UserMessage(
                role="user",
                content=USER_PROMPT.format(
                    user_instructions=user_instructions,
                    conversation=conversation_str,
                ),
            ),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_EVAL_USER_SIMULATOR,
            messages=messages,
            call_name="llm_judge_agent_hallucination_check",
        )

        total_agent = _count_agent_utterances(full_trajectory)

        try:
            reasoning, errors, summary = _parse_response(assistant_message.content)
            rate = (len(errors) / total_agent) if total_agent > 0 else 0.0
            return AgentHallucinationCheck(
                reasoning=reasoning,
                hallucination_found=len(errors) > 0,
                hallucination_rate=rate,
                total_agent_utterances=total_agent,
                errors=errors,
                summary=summary,
                cost=assistant_message.cost,
            )
        except Exception as e:
            return AgentHallucinationCheck(
                reasoning=f"Failed to parse LLM response: {e}",
                hallucination_found=False,
                hallucination_rate=0.0,
                total_agent_utterances=total_agent,
                errors=[
                    AgentHallucinationError(
                        reasoning=(
                            f"Failed to parse LLM response: {e}. "
                            f"Response: {assistant_message.content[:500]}"
                        ),
                    )
                ],
                summary="",
                cost=assistant_message.cost,
            )
