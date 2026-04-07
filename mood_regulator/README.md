# MoodRegulatorEnv

An OpenEnv-compatible environment where an AI agent tracks a user's emotional state over time and recommends the right content to improve or amplify their mood.

**Key insight:** When a user is sad, the agent comforts. When the user is happy, the agent motivates and energizes — not calms.

---

## What it does

The environment simulates a user whose mood shifts over time. The agent observes mood signals and recommends content (music, video, articles, activities, quotes) to improve or amplify the user's emotional state.

A real-time UI lets you type how you feel, detect your mood via LLM, and watch the agent respond step by step.

---

## Baseline scores

Run with seed=42 for reproducibility:

```
easy   → 0.80
medium → 1.00
hard   → 1.00
Overall → 0.93
```

---

## Action space

| Field | Type | Values |
|-------|------|--------|
| `content_type` | string | `music`, `video`, `article`, `activity`, `quote` |
| `mood_target` | string | `comfort`, `calm`, `distract`, `inspire`, `motivate`, `energize` |
| `reason` | string | Optional explanation |

---

## Observation space

| Field | Type | Description |
|-------|------|-------------|
| `current_mood` | string | `sad`, `anxious`, `angry`, `stressed`, `neutral`, `happy` |
| `mood_intensity` | float 0–1 | How strongly the user feels it |
| `mood_history` | list | Last 5 moods — use this to detect trends |
| `last_reaction` | string | `liked`, `skipped`, `engaged`, `ignored` |
| `session_step` | int | Steps elapsed |
| `session_over` | bool | True when episode ends |

---

## Tasks

| Task | Difficulty | Steps | Description |
|------|-----------|-------|-------------|
| `easy` | Easy | 5 | User is stably sad. Agent must comfort and improve mood. |
| `medium` | Medium | 8 | User starts neutral, shifts to stressed mid-session. Agent must detect and adapt. |
| `hard` | Hard | 12 | User starts happy, crashes to anxious, must recover to happy. |

---

## Reward function

Partial rewards at every step — not just end of episode:

- `+0.4` — correct mood target for current mood
- `+0.2` — correct content type for the target
- `+0.3` — mood actually improved after the action
- `±0.1` — user reaction bonus/penalty
- `-0.15` — intensity penalty for wrong action on high-intensity mood

---

## Setup

### Local

```bash
git clone <your-repo>
cd mood-regulator

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start server
uvicorn mood_regulator.main:app --reload --port 8000
```

Open `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the API explorer.

### Docker

```bash
docker build -t mood-regulator .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
  -e OPENAI_MODEL=llama-3.1-8b-instant \
  mood-regulator
```

---

## Run baseline agent

```bash
python -m mood_regulator.baseline_agent
```

Uses OpenAI-compatible client pointed at Groq. Set these environment variables:

```
OPENAI_API_KEY=your_groq_key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-8b-instant
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start a new session |
| GET | `/state` | Current mood state |
| POST | `/step` | Agent takes an action |
| GET | `/score` | Session score |
| POST | `/grade` | Detailed grader breakdown |
| POST | `/detect` | Detect mood from free text |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) |
| `OPENAI_API_KEY` | Same as GROQ_API_KEY (OpenAI-compatible) |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | `llama-3.1-8b-instant` |

---

## Project structure

```
mood-regulator/
├── main.py                  # FastAPI server
├── requirements.txt
├── Dockerfile
├── openenv.yaml             # OpenEnv spec
├── README.md
└── mood_regulator/
    ├── __init__.py
    ├── models.py            # Typed Pydantic models
    ├── environment.py       # step() / reset() / state()
    ├── reward.py            # Partial reward function
    ├── tasks.py             # 3 task definitions
    ├── graders.py           # 3 grader functions
    ├── baseline_agent.py    # LLM baseline agent
    ├── mood_detector.py     # Groq mood detection
    └── index.html           # Web UI
```