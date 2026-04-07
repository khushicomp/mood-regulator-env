"""
models.py — Data shapes for MoodRegulatorEnv

These are the typed models that define what the agent sees (State),
what the agent can do (Action), and what the environment returns (StepResult).
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ── Mood Types ────────────────────────────────────────────────────────────────

MoodType = Literal["sad", "anxious", "angry", "stressed", "neutral", "happy"]

ContentType = Literal["music", "article", "video", "activity", "quote"]

MoodTarget = Literal["comfort", "calm", "distract", "inspire", "motivate", "energize"]

Reaction = Literal["liked", "skipped", "engaged", "ignored"]


# ── State: What the agent observes ───────────────────────────────────────────

class MoodState(BaseModel):
    """
    Everything the agent can see about the current user session.

    - current_mood     : The user's mood right now
    - mood_intensity   : How strongly they feel it (0.0 = mild, 1.0 = extreme)
    - mood_history     : Last 5 moods (oldest → newest), so agent can spot trends
    - last_reaction    : How the user reacted to the last recommendation (None at start)
    - session_step     : How many steps have happened in this session
    - session_over     : True when the session has ended
    """
    current_mood: MoodType
    mood_intensity: float = Field(ge=0.0, le=1.0)
    mood_history: list[MoodType] = Field(default_factory=list, max_length=5)
    last_reaction: Reaction | None = None
    session_step: int = 0
    session_over: bool = False


# ── Action: What the agent can do ────────────────────────────────────────────

class Action(BaseModel):
    """
    The agent's single action type: recommend content.

    - content_type  : What kind of content to serve (music, video, etc.)
    - mood_target   : What emotional outcome the agent is aiming for
    - reason        : Optional — agent explains its reasoning (used in grading)

    Example:
        Action(content_type="music", mood_target="calm", reason="User is anxious")
    """
    content_type: ContentType
    mood_target: MoodTarget
    reason: str = ""


# ── StepResult: What the environment returns after each action ────────────────

class StepResult(BaseModel):
    """
    Returned by env.step(action).

    - state   : The new state after the action
    - reward  : Float 0.0–1.0 indicating how good the action was
    - done    : True if the session is over
    - info    : Extra diagnostic info (why reward was given, what changed)
    """
    state: MoodState
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict = Field(default_factory=dict)