"""
baseline_agent.py — LLM-powered baseline agent for MoodRegulatorEnv

Uses the OpenAI-compatible client pointed at Groq.
Reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.

Usage:
    python -m mood_regulator.baseline_agent
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import OpenAI
from .environment import MoodRegulatorEnv
from .models import Action

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("GROQ_API_KEY", "")),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
)

MODEL = os.environ.get("OPENAI_MODEL", "llama-3.1-8b-instant")

VALID_CONTENT = {"music", "article", "video", "activity", "quote"}
VALID_TARGETS = {"calm", "motivate", "comfort", "distract", "inspire", "energize"}

# Strict mood-to-action mapping to guide the LLM
MOOD_RULES = {
    "sad":     ("music",    "comfort"),
    "anxious": ("article",  "calm"),
    "angry":   ("activity", "distract"),
    "stressed":("music",    "calm"),
    "neutral": ("video",    "inspire"),
    "happy":   ("video",    "motivate"),
}

AGENT_SYSTEM_PROMPT = """
You are a mood-aware content recommendation agent.

Given the user's current mood state, recommend the best content to improve or amplify their mood.

You must respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

JSON format:
{
  "content_type": "<one of: music, video, article, activity, quote>",
  "mood_target": "<one of: calm, motivate, comfort, distract, inspire, energize>",
  "reason": "<one short sentence>"
}

STRICT RULES — follow these exactly:
- sad      → music    / comfort   (soothe the pain first)
- anxious  → article  / calm      (calm the racing thoughts)
- angry    → activity / distract  (redirect the energy)
- stressed → music    / calm      (reduce overwhelm)
- neutral  → video    / inspire   (spark engagement)
- happy    → video    / motivate  (amplify the positive energy!)

ADAPTATION RULES:
- If last_reaction is "skipped" or "ignored" 2+ times in a row → SWITCH content_type
- If mood is WORSENING (check mood_history) → switch to the correct mood rule above immediately
- If mood is IMPROVING → keep the same strategy

NEVER recommend "inspire" for sad, angry, anxious, or stressed users.
NEVER recommend "motivate" for sad, angry, anxious, or stressed users.
"""


def call_agent(mood_state: dict) -> dict:
    """Ask the LLM what action to take given the current mood state."""

    # Build a clear, structured prompt
    user_message = f"""
Current mood state:
- Current mood  : {mood_state['current_mood']}
- Intensity     : {mood_state['mood_intensity']}
- Mood history  : {mood_state['mood_history']}
- Last reaction : {mood_state['last_reaction']}
- Step number   : {mood_state['session_step']}

Based on the STRICT RULES, what is the correct content_type and mood_target for mood "{mood_state['current_mood']}"?
Respond with ONLY the JSON object.
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=150,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low temperature = more deterministic
    )

    text = response.choices[0].message.content.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text)


def run_task(task: str, seed: int = 42) -> float:
    """Run a single task with the LLM agent and return the grade score."""
    env = MoodRegulatorEnv(task=task, seed=seed)
    state = env.reset()

    print(f"\n{'='*55}")
    print(f"  TASK: {task.upper()}  |  Starting mood: {state.current_mood}  |  Intensity: {state.mood_intensity:.2f}")
    print(f"{'='*55}")

    step_num = 0

    while not state.session_over:
        step_num += 1

        mood_dict = {
            "current_mood": state.current_mood,
            "mood_intensity": state.mood_intensity,
            "mood_history": state.mood_history,
            "last_reaction": state.last_reaction,
            "session_step": state.session_step,
        }

        try:
            decision = call_agent(mood_dict)
            content_type = decision.get("content_type", "video")
            mood_target = decision.get("mood_target", "inspire")
            reason = decision.get("reason", "LLM decision")

            # Validate
            if content_type not in VALID_CONTENT:
                content_type = MOOD_RULES.get(state.current_mood, ("video", "inspire"))[0]
            if mood_target not in VALID_TARGETS:
                mood_target = MOOD_RULES.get(state.current_mood, ("video", "inspire"))[1]

        except Exception as e:
            # Hard fallback — use mood rules directly
            content_type, mood_target = MOOD_RULES.get(state.current_mood, ("video", "inspire"))
            reason = f"fallback rule: {e}"

        action = Action(
            content_type=content_type,
            mood_target=mood_target,
            reason=reason,
        )

        prev_mood = state.current_mood
        result = env.step(action)
        state = result.state

        print(
            f"  Step {step_num:02d} | "
            f"Mood: {prev_mood:>8} → {state.current_mood:<8} | "
            f"Action: {content_type:<8} / {mood_target:<9} | "
            f"Reward: {result.reward:.2f} | "
            f"Reaction: {result.state.last_reaction}"
        )

    from .graders import grade
    grade_score, grade_reason = grade(
        task_name=task,
        reward_history=env._reward_history,
        mood_history=env._state.mood_history,
        final_mood=state.current_mood,
    )

    print(f"\n  Final mood    : {state.current_mood}")
    print(f"  Session score : {env.session_score():.2f}")
    print(f"  Grade score   : {grade_score:.2f}")
    print(f"  Grade reason  : {grade_reason}")

    return grade_score


def main():
    print("\n" + "=" * 55)
    print("  🧠 MoodRegulatorEnv — LLM Baseline Agent")
    print("  Model : " + MODEL)
    print("  Seed  : 42 (fixed for reproducibility)")
    print("=" * 55)

    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        scores[task] = run_task(task, seed=42)

    print(f"\n{'='*55}")
    print(f"  FINAL SCORES")
    print(f"{'='*55}")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<8} : {score:.2f}  {bar}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Overall avg : {avg:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()