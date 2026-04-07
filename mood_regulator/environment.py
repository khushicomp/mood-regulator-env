"""
environment.py — MoodRegulatorEnv core engine

Implements the 3 required OpenEnv methods:
    - reset()  : Start a fresh session
    - state()  : Return current state
    - step()   : Agent takes action, environment responds
"""

from __future__ import annotations
import random
from .models import MoodState, Action, StepResult, MoodType, Reaction
from .reward import compute_reward, mood_improved

MOOD_LADDER: list[MoodType] = [
    "angry", "sad", "anxious", "stressed", "neutral", "happy"
]


def _shift_mood(
    current: MoodType,
    reward: float,
    intensity: float,
    rng: random.Random,
) -> tuple[MoodType, float]:
    idx = MOOD_LADDER.index(current)

    if reward >= 0.7:
        shift = rng.choices([-0, 1, 2], weights=[0.1, 0.6, 0.3])[0]
    elif reward >= 0.4:
        shift = rng.choices([-1, 0, 1], weights=[0.2, 0.5, 0.3])[0]
    else:
        shift = rng.choices([-2, -1, 0], weights=[0.2, 0.5, 0.3])[0]

    new_idx = max(0, min(len(MOOD_LADDER) - 1, idx + shift))
    new_mood: MoodType = MOOD_LADDER[new_idx]

    if new_idx > idx:
        new_intensity = max(0.1, intensity - rng.uniform(0.1, 0.2))
    elif new_idx < idx:
        new_intensity = min(1.0, intensity + rng.uniform(0.1, 0.2))
    else:
        new_intensity = intensity

    return new_mood, round(new_intensity, 2)


def _simulate_reaction(reward: float, rng: random.Random) -> Reaction:
    """
    Simulate user reaction. Includes intentional noise to prevent
    simple exploits — even good actions occasionally get skipped,
    and bad actions occasionally get engaged with.
    """
    if reward >= 0.7:
        return rng.choices(
            ["liked", "engaged", "skipped", "ignored"],
            weights=[0.45, 0.35, 0.15, 0.05]
        )[0]  # type: ignore
    elif reward >= 0.4:
        return rng.choices(
            ["engaged", "skipped", "ignored", "liked"],
            weights=[0.35, 0.35, 0.20, 0.10]
        )[0]  # type: ignore
    else:
        return rng.choices(
            ["skipped", "ignored", "engaged", "liked"],
            weights=[0.45, 0.35, 0.15, 0.05]
        )[0]  # type: ignore


class MoodRegulatorEnv:
    """
    OpenEnv-compatible environment for mood-based content recommendation.

    The agent observes a user's mood state and recommends content to
    improve or amplify their mood. Sessions run for `max_steps` turns.

    Usage:
        env = MoodRegulatorEnv(task="easy", seed=42)
        state = env.reset()

        while not state.session_over:
            action = agent.act(state)
            result = env.step(action)
            state = result.state

        score = env.session_score()
    """

    MAX_STEPS_PER_TASK = {
        "easy":   5,
        "medium": 8,
        "hard":   12,
    }

    TASK_CONFIGS = {
        "easy": {
            "starting_moods": ["sad", "anxious", "happy"],
            "mood_drift": False,
        },
        "medium": {
            "starting_moods": ["sad", "neutral", "stressed"],
            "mood_drift": True,
        },
        "hard": {
            "starting_moods": ["angry", "anxious", "sad"],
            "mood_drift": True,
        },
    }

    def __init__(self, task: str = "easy", seed: int | None = None):
        assert task in self.TASK_CONFIGS, f"task must be one of {list(self.TASK_CONFIGS)}"
        self.task = task
        self.max_steps = self.MAX_STEPS_PER_TASK[task]
        self.seed = seed

        # Use a dedicated RNG instance — fully isolated, reproducible
        self._rng = random.Random(seed)

        self._state: MoodState | None = None
        self._reward_history: list[float] = []

    def reset(self) -> MoodState:
        """
        Start a fresh session. Returns the initial state.
        Re-seeds the RNG so every reset() with the same seed
        produces identical trajectories — fully reproducible.
        """
        # Re-seed on every reset for full reproducibility
        self._rng = random.Random(self.seed)

        config = self.TASK_CONFIGS[self.task]
        starting_mood: MoodType = self._rng.choice(config["starting_moods"])
        starting_intensity = round(self._rng.uniform(0.4, 0.9), 2)

        self._state = MoodState(
            current_mood=starting_mood,
            mood_intensity=starting_intensity,
            mood_history=[],
            last_reaction=None,
            session_step=0,
            session_over=False,
        )
        self._reward_history = []
        return self._state

    def state(self) -> MoodState:
        """Return current state without modifying anything."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    def step(self, action: Action) -> StepResult:
        """
        Agent takes an action. Environment responds with new state + reward.

        Returns StepResult with:
            - state  : new MoodState (observation)
            - reward : float 0.0-1.0
            - done   : True if session is over
            - info   : diagnostic dict
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.session_over:
            raise RuntimeError("Session is over. Call reset() to start a new one.")

        current_mood = self._state.current_mood
        current_intensity = self._state.mood_intensity

        # 1. Compute initial reward
        reward, reward_reason = compute_reward(
            mood=current_mood,
            mood_intensity=current_intensity,
            content_type=action.content_type,
            mood_target=action.mood_target,
            last_reaction=self._state.last_reaction,
            mood_improved=False,
        )

        # 2. Simulate user reaction (uses isolated RNG — reproducible)
        reaction: Reaction = _simulate_reaction(reward, self._rng)

        # 3. Shift mood based on reward
        new_mood, new_intensity = _shift_mood(
            current_mood, reward, current_intensity, self._rng
        )

        # 4. Recompute reward with mood_improved factored in
        improved = mood_improved(current_mood, new_mood)
        reward, reward_reason = compute_reward(
            mood=current_mood,
            mood_intensity=current_intensity,
            content_type=action.content_type,
            mood_target=action.mood_target,
            last_reaction=self._state.last_reaction,
            mood_improved=improved,
        )

        self._reward_history.append(reward)

        # 5. Update state
        new_history = (self._state.mood_history + [current_mood])[-5:]
        new_step = self._state.session_step + 1
        done = new_step >= self.max_steps

        self._state = MoodState(
            current_mood=new_mood,
            mood_intensity=new_intensity,
            mood_history=new_history,
            last_reaction=reaction,
            session_step=new_step,
            session_over=done,
        )

        return StepResult(
            state=self._state,
            reward=reward,
            done=done,
            info={
                "action_taken": action.model_dump(),
                "mood_before": current_mood,
                "mood_after": new_mood,
                "mood_improved": improved,
                "user_reaction": reaction,
                "reward_reason": reward_reason,
            }
        )

    def session_score(self) -> float:
        """Final score 0.0-1.0. Average reward + happiness bonus."""
        if not self._reward_history:
            return 0.0
        avg_reward = sum(self._reward_history) / len(self._reward_history)
        final_mood = self._state.current_mood if self._state else "neutral"
        happiness_bonus = 0.1 if final_mood == "happy" else 0.0
        return round(min(1.0, avg_reward + happiness_bonus), 2)