"""
graders.py — Agent graders for MoodRegulatorEnv

Each task has its own grader function that scores the agent's
performance from 0.0 to 1.0.

A grader receives:
    - reward_history  : list of rewards per step
    - mood_history    : list of moods across the session
    - final_mood      : the user's mood at session end
    - task            : the Task object being evaluated

And returns a score between 0.0 and 1.0.
"""

from .tasks import Task, TASKS


# ── Grader: Easy Task ─────────────────────────────────────────────────────────

def grade_easy(
    reward_history: list[float],
    mood_history: list[str],
    final_mood: str,
    task: Task,
) -> tuple[float, str]:
    """
    Grade the easy task: stable sad user.

    Scoring:
        0.4  → Final mood reached target (neutral or happy)
        0.3  → Average reward >= 0.5
        0.2  → Mood improved at least once during session
        0.1  → Agent used at least 3 steps (didn't give up early)

    Max score: 1.0
    """
    score = 0.0
    reasons = []

    criteria = task.success_criteria
    target_moods = criteria["target_moods"]
    min_avg_reward = criteria["min_avg_reward"]

    # Check 1: Did mood reach target?
    if final_mood in target_moods:
        score += 0.4
        reasons.append(f"✅ Final mood '{final_mood}' reached target")
    else:
        reasons.append(f"❌ Final mood '{final_mood}' did not reach target {target_moods}")

    # Check 2: Was average reward good enough?
    avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
    if avg_reward >= min_avg_reward:
        score += 0.3
        reasons.append(f"✅ Avg reward {avg_reward:.2f} >= {min_avg_reward}")
    else:
        reasons.append(f"❌ Avg reward {avg_reward:.2f} < {min_avg_reward}")

    # Check 3: Did mood improve at least once?
    from .reward import MOOD_VALENCE
    mood_values = [MOOD_VALENCE.get(m, 0) for m in mood_history]  # type: ignore
    if len(mood_values) >= 2 and max(mood_values) > mood_values[0]:
        score += 0.2
        reasons.append("✅ Mood improved at least once during session")
    else:
        reasons.append("❌ Mood never improved during session")

    # Check 4: Did the agent use enough steps?
    if len(reward_history) >= 3:
        score += 0.1
        reasons.append(f"✅ Agent used {len(reward_history)} steps")
    else:
        reasons.append(f"❌ Agent only used {len(reward_history)} steps")

    return round(score, 2), " | ".join(reasons)


# ── Grader: Medium Task ───────────────────────────────────────────────────────

def grade_medium(
    reward_history: list[float],
    mood_history: list[str],
    final_mood: str,
    task: Task,
) -> tuple[float, str]:
    """
    Grade the medium task: shifting mood user.

    The key extra check: did the agent ADAPT when the mood shifted?
    We detect this by checking if the reward stayed healthy AFTER
    the mood changed (rewards shouldn't crash after a mood shift).

    Scoring:
        0.3  → Final mood reached target
        0.3  → Avg reward >= 0.5
        0.2  → Reward stayed above 0.4 after mood shift (adaptation detected)
        0.2  → Mood history shows improvement trend in second half
    """
    score = 0.0
    reasons = []

    criteria = task.success_criteria
    target_moods = criteria["target_moods"]
    min_avg_reward = criteria["min_avg_reward"]

    # Check 1: Final mood
    if final_mood in target_moods:
        score += 0.3
        reasons.append(f"✅ Final mood '{final_mood}' reached target")
    else:
        reasons.append(f"❌ Final mood '{final_mood}' did not reach target")

    # Check 2: Average reward
    avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
    if avg_reward >= min_avg_reward:
        score += 0.3
        reasons.append(f"✅ Avg reward {avg_reward:.2f} >= {min_avg_reward}")
    else:
        reasons.append(f"❌ Avg reward {avg_reward:.2f} < {min_avg_reward}")

    # Check 3: Did agent adapt to mood shift?
    # We look at rewards in the second half of the session
    # If agent adapted, rewards in second half should be >= 0.4
    if len(reward_history) >= 4:
        midpoint = len(reward_history) // 2
        second_half_rewards = reward_history[midpoint:]
        avg_second_half = sum(second_half_rewards) / len(second_half_rewards)
        if avg_second_half >= 0.4:
            score += 0.2
            reasons.append(f"✅ Agent adapted: second-half avg reward {avg_second_half:.2f}")
        else:
            reasons.append(f"❌ Agent failed to adapt: second-half avg reward {avg_second_half:.2f}")
    else:
        reasons.append("❌ Not enough steps to evaluate adaptation")

    # Check 4: Mood trend in second half
    from .reward import MOOD_VALENCE
    if len(mood_history) >= 4:
        midpoint = len(mood_history) // 2
        first_half_avg = sum(MOOD_VALENCE.get(m, 0) for m in mood_history[:midpoint]) / midpoint  # type: ignore[call-overload]
        second_half_avg = sum(MOOD_VALENCE.get(m, 0) for m in mood_history[midpoint:]) / (len(mood_history) - midpoint)  # type: ignore[call-overload]
        if second_half_avg >= first_half_avg:
            score += 0.2
            reasons.append("✅ Mood trend improved in second half of session")
        else:
            reasons.append("❌ Mood trend worsened in second half")
    else:
        reasons.append("❌ Not enough mood history to evaluate trend")

    return round(score, 2), " | ".join(reasons)


# ── Grader: Hard Task ─────────────────────────────────────────────────────────

def grade_hard(
    reward_history: list[float],
    mood_history: list[str],
    final_mood: str,
    task: Task,
) -> tuple[float, str]:
    """
    Grade the hard task: happy → crash → recovery.

    This grader checks for the full arc:
        Phase 1 (steps 1-4)  : Agent amplifies happy mood correctly
        Phase 2 (steps 5-8)  : Agent detects crash and switches strategy
        Phase 3 (steps 9-12) : Agent recovers mood to neutral/happy

    Scoring:
        0.2  → Phase 1: Avg reward >= 0.6 while mood was happy
        0.3  → Phase 2: Reward didn't collapse after crash (>= 0.4 avg)
        0.3  → Phase 3: Final mood is neutral or happy
        0.2  → Overall avg reward >= 0.6
    """
    score = 0.0
    reasons = []

    criteria = task.success_criteria
    target_moods = criteria["target_moods"]
    min_avg_reward = criteria["min_avg_reward"]

    # Split session into 3 phases
    n = len(reward_history)
    phase1 = reward_history[:n//3]
    phase2 = reward_history[n//3: 2*n//3]
    phase3 = reward_history[2*n//3:]

    # Check 1: Phase 1 — did agent amplify happy mood?
    if phase1:
        avg_p1 = sum(phase1) / len(phase1)
        if avg_p1 >= 0.6:
            score += 0.2
            reasons.append(f"✅ Phase 1: amplified happy mood well (avg {avg_p1:.2f})")
        else:
            reasons.append(f"❌ Phase 1: poor amplification of happy mood (avg {avg_p1:.2f})")

    # Check 2: Phase 2 — did agent recover after crash?
    if phase2:
        avg_p2 = sum(phase2) / len(phase2)
        if avg_p2 >= 0.4:
            score += 0.3
            reasons.append(f"✅ Phase 2: adapted after mood crash (avg {avg_p2:.2f})")
        else:
            reasons.append(f"❌ Phase 2: failed to adapt after crash (avg {avg_p2:.2f})")

    # Check 3: Phase 3 — final mood
    if final_mood in target_moods:
        score += 0.3
        reasons.append(f"✅ Phase 3: recovered to '{final_mood}'")
    else:
        reasons.append(f"❌ Phase 3: ended on '{final_mood}', not recovered")

    # Check 4: Overall avg reward
    avg_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
    if avg_reward >= min_avg_reward:
        score += 0.2
        reasons.append(f"✅ Overall avg reward {avg_reward:.2f} >= {min_avg_reward}")
    else:
        reasons.append(f"❌ Overall avg reward {avg_reward:.2f} < {min_avg_reward}")

    return round(score, 2), " | ".join(reasons)


# ── Unified grade() function ──────────────────────────────────────────────────

def grade(
    task_name: str,
    reward_history: list[float],
    mood_history: list[str],
    final_mood: str,
) -> tuple[float, str]:
    """
    Main entry point for grading. Routes to the correct grader by task name.

    Args:
        task_name      : "easy" | "medium" | "hard"
        reward_history : list of rewards per step
        mood_history   : list of moods across session
        final_mood     : mood at session end

    Returns:
        (score: float 0.0–1.0, explanation: str)
    """
    task = TASKS[task_name]

    if task_name == "easy":
        return grade_easy(reward_history, mood_history, final_mood, task)
    elif task_name == "medium":
        return grade_medium(reward_history, mood_history, final_mood, task)
    elif task_name == "hard":
        return grade_hard(reward_history, mood_history, final_mood, task)
    else:
        raise ValueError(f"Unknown task: {task_name}")