"""
main.py — FastAPI server for MoodRegulatorEnv

Exposes the environment as an HTTP API so any agent can interact with it.

Endpoints:
    POST /reset          → Start a new session
    GET  /state          → Observe current state
    POST /step           → Take an action
    GET  /score          → Get session score
    POST /grade          → Grade a completed session
    GET  /tasks          → List available tasks
    GET  /health         → Health check
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from .environment import MoodRegulatorEnv
from .models import Action, MoodState, StepResult
from .graders import grade
from .tasks import ALL_TASKS
from .mood_detector import MoodDetector
from dotenv import load_dotenv
load_dotenv()
# One detector per server (maintains message history across /detect calls)
detector = MoodDetector()


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MoodRegulatorEnv",
    description=(
        "An OpenEnv-compatible environment where an AI agent tracks a user's "
        "mood over time and recommends the right content to improve or amplify it."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (stateful)
# For production: use session IDs + a dict of envs per user
env: MoodRegulatorEnv | None = None


# ── Request / Response Models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"      # "easy" | "medium" | "hard"
    seed: int | None = None  # optional fixed seed for reproducibility

class StepRequest(BaseModel):
    content_type: str       # "music" | "article" | "video" | "activity" | "quote"
    mood_target: str        # "comfort" | "calm" | "distract" | "inspire" | "motivate" | "energize"
    reason: str = ""        # optional explanation from agent

class GradeRequest(BaseModel):
    task_name: str
    reward_history: list[float]
    mood_history: list[str]
    final_mood: str

class DetectRequest(BaseModel):
    text: str       # anything the user typed
    reset_history: bool = False  # set True to start fresh detection session


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple health check — confirms the server is running."""
    return {"status": "ok", "environment": "MoodRegulatorEnv"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with their descriptions."""
    return {
        "tasks": [
            {
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
                "success_criteria": t.success_criteria,
            }
            for t in ALL_TASKS
        ]
    }

@app.get("/")
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(ui_path)
@app.post("/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    """
    Start a fresh session.

    Body:
        task  : Difficulty level ("easy" | "medium" | "hard")
        seed  : Optional integer for reproducible sessions

    Returns:
        Initial MoodState
    """
    global env

    valid_tasks = ["easy", "medium", "hard"]
    if request.task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{request.task}'. Must be one of {valid_tasks}"
        )

    env = MoodRegulatorEnv(task=request.task, seed=request.seed)
    initial_state = env.reset()

    return {
        "message": f"Session started — task: {request.task}",
        "state": initial_state.model_dump(),
    }


@app.get("/state")
def get_state() -> dict[str, Any]:
    """
    Observe the current state without changing anything.

    Returns:
        Current MoodState
    """
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first."
        )
    return {"state": env.state().model_dump()}


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    """
    Agent takes an action. Environment responds with new state + reward.

    Body:
        content_type : What to recommend ("music" | "article" | "video" | "activity" | "quote")
        mood_target  : Intended emotional outcome
        reason       : Optional — agent's reasoning

    Returns:
        StepResult with new state, reward, done, info
    """
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first."
        )

    # Validate action fields
    valid_content = ["music", "article", "video", "activity", "quote"]
    valid_targets = ["comfort", "calm", "distract", "inspire", "motivate", "energize"]

    if request.content_type not in valid_content:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content_type '{request.content_type}'. Must be one of {valid_content}"
        )
    if request.mood_target not in valid_targets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mood_target '{request.mood_target}'. Must be one of {valid_targets}"
        )

    action = Action(
        content_type=request.content_type, # type: ignore
        mood_target=request.mood_target, # type: ignore
        reason=request.reason,
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "state": result.state.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/score")
def get_score() -> dict[str, Any]:
    """
    Get the final session score (0.0 – 1.0).
    Call this after the session is done (done=True from /step).

    Returns:
        session_score : float
    """
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first."
        )
    return {
        "session_score": env.session_score(),
        "reward_history": env._reward_history,
    }


@app.post("/grade")
def grade_session(request: GradeRequest) -> dict[str, Any]:
    """
    Grade a completed session using the task-specific grader.

    Body:
        task_name      : "easy" | "medium" | "hard"
        reward_history : List of rewards per step
        mood_history   : List of moods across session
        final_mood     : Mood at session end

    Returns:
        score       : float 0.0–1.0
        explanation : Breakdown of how the score was computed
    """
    valid_tasks = ["easy", "medium", "hard"]
    if request.task_name not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_name. Must be one of {valid_tasks}"
        )

    score, explanation = grade(
        task_name=request.task_name,
        reward_history=request.reward_history,
        mood_history=request.mood_history,
        final_mood=request.final_mood,
    )

    return {
        "task": request.task_name,
        "score": score,
        "explanation": explanation,
    }


@app.post("/detect")
def detect_mood(request: DetectRequest) -> dict[str, Any]:
    """
    Detect mood from free-form user text using Claude.

    This is the REAL entry point for a live user session.
    Instead of simulating mood, it detects it from what the user actually types.

    Body:
        text          : Anything the user typed ("I feel exhausted today")
        reset_history : Set True to clear previous message history

    Returns:
        MoodState — plug this directly into /reset or use as env observation

    Full pipeline example:
        1. POST /detect  {"text": "I'm so stressed with deadlines"}
           → returns MoodState(current_mood="stressed", mood_intensity=0.8)
        2. POST /reset   {"task": "medium"}
           → start environment session
        3. GET  /state   → agent reads state
        4. POST /step    → agent recommends content
    """
    if request.reset_history:
        detector.reset()

    try:
        mood_state = detector.detect(request.text)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mood detection failed: {str(e)}")

    return {
        "detected": mood_state.model_dump(),
        "message": (
            f"Detected mood: {mood_state.current_mood} "
            f"(intensity: {mood_state.mood_intensity})"
        ),
    }