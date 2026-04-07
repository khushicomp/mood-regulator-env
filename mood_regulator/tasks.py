"""
tasks.py — Task definitions for MoodRegulatorEnv

Each task defines a scenario the agent must solve.
Tasks increase in difficulty: easy → medium → hard

A task defines:
    - name        : Unique identifier
    - difficulty  : "easy" | "medium" | "hard"
    - description : What the agent must do
    - starting_mood    : What mood the user starts with
    - success_criteria : What the grader checks for
    - max_steps   : How many turns the agent gets
"""

from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    difficulty: str
    description: str
    starting_mood: str
    success_criteria: dict
    max_steps: int
    seed: int = 42              # Fixed seed = reproducible results for grading


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — EASY
# The user has a single stable sad mood.
# The agent must correctly comfort them and improve their mood.
# No surprises — just get the basics right.
# ─────────────────────────────────────────────────────────────────────────────

TASK_EASY = Task(
    name="stable_sad_user",
    difficulty="easy",
    description=(
        "The user is feeling sad with moderate intensity. "
        "Their mood is stable — it won't shift unexpectedly. "
        "The agent must detect the sad mood and serve comforting or "
        "uplifting content to improve it. "
        "Success = mood improves to neutral or happy within 5 steps."
    ),
    starting_mood="sad",
    success_criteria={
        "target_moods": ["neutral", "happy"],   # mood must reach one of these
        "min_avg_reward": 0.5,                  # average reward must be at least 0.5
        "max_steps_allowed": 5,
    },
    max_steps=5,
    seed=42,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM
# The user starts neutral but shifts to stressed mid-session.
# The agent must DETECT the shift and adapt its recommendations.
# Wrong: keep recommending energizing content after mood drops to stressed.
# Right: switch to calming content when stress appears.
# ─────────────────────────────────────────────────────────────────────────────

TASK_MEDIUM = Task(
    name="shifting_mood_user",
    difficulty="medium",
    description=(
        "The user starts neutral but their mood shifts to stressed around step 3. "
        "The agent must notice this shift (via mood_history and current_mood) "
        "and adapt its recommendations accordingly. "
        "Serving energizing content to a stressed user will be penalized. "
        "Success = agent adapts within 2 steps of the mood shift "
        "and ends session at neutral or better."
    ),
    starting_mood="neutral",
    success_criteria={
        "target_moods": ["neutral", "happy"],
        "min_avg_reward": 0.5,
        "mood_shift_detected": True,            # agent must change strategy after shift
        "max_steps_allowed": 8,
    },
    max_steps=8,
    seed=7,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — HARD
# The user starts happy. The agent must serve motivating/energizing content
# to amplify the good mood. Mid-session the user becomes anxious.
# The agent must detect this NEGATIVE shift (from happy → anxious),
# switch strategy entirely, stabilize the mood, then push back to happy.
# ─────────────────────────────────────────────────────────────────────────────

TASK_HARD = Task(
    name="happy_then_crash_recovery",
    difficulty="hard",
    description=(
        "The user starts happy. The agent should amplify this with motivating "
        "and energizing content. However, mid-session the user's mood crashes "
        "to anxious. The agent must: "
        "(1) Detect the crash via mood_history, "
        "(2) Switch from motivate/energize to calm/distract immediately, "
        "(3) Stabilize the mood, "
        "(4) Gradually push back toward happy. "
        "Success = agent recovers the mood to neutral or happy "
        "within 12 steps after a crash, with avg reward >= 0.6."
    ),
    starting_mood="happy",
    success_criteria={
        "target_moods": ["neutral", "happy"],
        "min_avg_reward": 0.6,                  # higher bar for hard task
        "must_recover_from_crash": True,        # mood must crash AND recover
        "max_steps_allowed": 12,
    },
    max_steps=12,
    seed=13,
)


# ─────────────────────────────────────────────────────────────────────────────
# TASK REGISTRY — import this in other files
# ─────────────────────────────────────────────────────────────────────────────

TASKS: dict[str, Task] = {
    "easy":   TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard":   TASK_HARD,
}

ALL_TASKS = list(TASKS.values())