"""
reward.py — Reward logic for MoodRegulatorEnv

The reward function answers: "How good was the agent's action?"
It gives PARTIAL rewards so the agent can learn incrementally.

Reward Scale:
    1.0  → Perfect match: right content, right target, mood improved
    0.7  → Good match: right target for mood, mood stayed stable
    0.4  → Partial match: content type was okay but target was off
    0.1  → Wrong match: content won't help this mood at all
    0.0  → Actively harmful: e.g. showing sad content to an already sad user
"""

from .models import MoodType, MoodTarget, ContentType, Reaction


# ── What each mood NEEDS ──────────────────────────────────────────────────────
#
# This is YOUR design decision — the mapping of mood → ideal targets.
# Key insight: happy users should get motivating/energizing content,
# not calming content (that would waste their positive energy).

IDEAL_TARGETS: dict[MoodType, list[MoodTarget]] = {
    "sad":      ["comfort", "inspire"],        # First comfort, then lift
    "anxious":  ["calm", "distract"],          # Reduce overwhelm first
    "angry":    ["distract", "calm"],          # Redirect, then soothe
    "stressed": ["calm", "distract"],          # Release tension
    "neutral":  ["inspire", "energize"],       # Keep momentum going
    "happy":    ["motivate", "energize"],      # Amplify the good mood! 🚀
}

# ── What content works for what target ───────────────────────────────────────
#
# Not all content types suit every mood target.
# E.g. "activity" is great for distraction but bad for comfort.

GOOD_CONTENT_FOR_TARGET: dict[MoodTarget, list[ContentType]] = {
    "comfort":   ["music", "quote", "article"],
    "calm":      ["music", "article", "activity"],
    "distract":  ["video", "activity", "music"],
    "inspire":   ["video", "article", "quote"],
    "motivate":  ["video", "article", "quote"],   # motivational content for happy users
    "energize":  ["music", "video", "activity"],  # high-energy content for happy users
}

# ── Reactions that signal the recommendation was good ────────────────────────

POSITIVE_REACTIONS: list[Reaction] = ["liked", "engaged"]
NEGATIVE_REACTIONS: list[Reaction] = ["skipped", "ignored"]


# ── Main Reward Function ──────────────────────────────────────────────────────

def compute_reward(
    mood: MoodType,
    mood_intensity: float,
    content_type: ContentType,
    mood_target: MoodTarget,
    last_reaction: Reaction | None,
    mood_improved: bool,
) -> tuple[float, str]:
    """
    Compute reward for the agent's action and return (reward, explanation).

    Parameters:
        mood           : User's current mood before the action
        mood_intensity : How strongly they feel it (0.0–1.0)
        content_type   : What the agent recommended
        mood_target    : What outcome the agent was aiming for
        last_reaction  : How the user reacted to the PREVIOUS recommendation
        mood_improved  : Did the mood get better after this action?

    Returns:
        (reward: float, reason: str)
    """

    reward = 0.0
    reasons = []

    # ── Step 1: Was the mood_target right for this mood? ──────────────────────
    ideal = IDEAL_TARGETS.get(mood, [])

    if mood_target in ideal:
        reward += 0.4
        reasons.append(f"✅ Good target '{mood_target}' for mood '{mood}'")
    else:
        reward += 0.1
        reasons.append(f"⚠️ Target '{mood_target}' is not ideal for mood '{mood}'")

    # ── Step 2: Was the content type right for the target? ────────────────────
    good_content = GOOD_CONTENT_FOR_TARGET.get(mood_target, [])

    if content_type in good_content:
        reward += 0.2
        reasons.append(f"✅ '{content_type}' fits target '{mood_target}'")
    else:
        reasons.append(f"⚠️ '{content_type}' is a weak choice for target '{mood_target}'")

    # ── Step 3: Did the mood actually improve? ────────────────────────────────
    if mood_improved:
        reward += 0.3
        reasons.append("✅ Mood improved after recommendation")
    else:
        reasons.append("➡️ Mood did not improve")

    # ── Step 4: Reaction bonus/penalty ───────────────────────────────────────
    if last_reaction in POSITIVE_REACTIONS:
        reward = min(1.0, reward + 0.1)
        reasons.append(f"✅ User reacted positively ({last_reaction})")
    elif last_reaction in NEGATIVE_REACTIONS:
        reward = max(0.0, reward - 0.1)
        reasons.append(f"❌ User reacted negatively ({last_reaction})")

    # ── Step 5: Intensity penalty — high intensity needs precise action ───────
    # If mood is very intense (0.8+) and the action was wrong, penalize harder
    if mood_intensity >= 0.8 and mood_target not in ideal:
        reward = max(0.0, reward - 0.15)
        reasons.append("❌ High intensity mood needs precise action — wrong target penalized")

    return round(reward, 2), " | ".join(reasons)


# ── Helper: Did mood improve? ─────────────────────────────────────────────────

MOOD_VALENCE: dict[MoodType, int] = {
    "angry":    0,
    "sad":      1,
    "anxious":  2,
    "stressed": 3,
    "neutral":  4,
    "happy":    5,
}

def mood_improved(before: MoodType, after: MoodType) -> bool:
    """Returns True if the mood moved in a positive direction."""
    return MOOD_VALENCE.get(after, 0) > MOOD_VALENCE.get(before, 0)