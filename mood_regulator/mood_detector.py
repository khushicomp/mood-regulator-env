from dotenv import load_dotenv # type: ignore
load_dotenv()

"""
mood_detector.py — LLM-based mood detection from user text

Takes any free-form user text and returns a MoodState.
Uses Groq (llama-3.1-8b-instant) to understand context, tone,
and subtle emotional signals that rule-based systems would miss.

Usage:
    detector = MoodDetector()
    state = detector.detect("I have so much work today, I can't breathe")
    # → MoodState(current_mood="stressed", mood_intensity=0.85, ...)
"""

import os
import json
import httpx # type: ignore
from .models import MoodState


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """
You are a mood detection system. Given a user's text, analyze their emotional state.

You must respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

JSON format:
{
  "mood": "<one of: sad, anxious, angry, stressed, neutral, happy>",
  "intensity": <float between 0.0 and 1.0>,
  "reasoning": "<one short sentence explaining your detection>"
}

Mood definitions:
- sad      : grief, loss, loneliness, hopelessness, crying
- anxious  : worry, fear, nervousness, overthinking, panic
- angry    : frustration, rage, irritation, injustice
- stressed : overwhelmed, too much to do, pressure, deadlines
- neutral  : calm, okay, neither good nor bad
- happy    : joy, excitement, contentment, positivity, pride

Intensity guide:
- 0.1–0.3 : mild (e.g. "a bit tired")
- 0.4–0.6 : moderate (e.g. "pretty stressed")
- 0.7–0.9 : strong (e.g. "extremely anxious")
- 1.0     : extreme (e.g. "I can't take this anymore")

Be sensitive to:
- Indirect expressions ("everything is fine I guess" → sad/neutral, low intensity)
- Sarcasm ("oh great, another Monday" → stressed/angry)
- Casual positivity ("just got promoted!!" → happy, high intensity)
"""


class MoodDetector:
    """
    Detects mood from free-form user text using Groq (llama-3.1-8b-instant).

    Maintains a short history of recent user messages to detect
    mood TRENDS — not just the current message in isolation.
    """

    def __init__(self):
        self._history: list[str] = []   # last 5 user messages

    def detect(self, user_text: str) -> MoodState:
        """
        Analyze user_text and return a MoodState.

        Args:
            user_text : Anything the user typed

        Returns:
            MoodState with current_mood, mood_intensity, mood_history
        """
        # Keep history of last 5 messages
        self._history = (self._history + [user_text])[-5:]

        # Build context — include history so model sees the trend
        history_context = ""
        if len(self._history) > 1:
            history_context = "\n\nPrevious messages (oldest → newest):\n"
            for msg in self._history[:-1]:
                history_context += f"  - {msg}\n"
            history_context += "\nCurrent message (analyze this primarily):"

        user_content = f"{history_context}\n{user_text}"

        # Call Groq API
        raw = self._call_groq(user_content)

        # Parse response
        mood = raw.get("mood", "neutral")
        intensity = float(raw.get("intensity", 0.5))
        intensity = round(max(0.0, min(1.0, intensity)), 2)

        # Build mood history from internal tracking
        mood_history = self._extract_mood_history()

        return MoodState(
            current_mood=mood,
            mood_intensity=intensity,
            mood_history=mood_history,  # type: ignore
            last_reaction=None,
            session_step=0,
            session_over=False,
        )

    def _call_groq(self, user_content: str) -> dict:
        """Call Groq API and return parsed JSON response."""
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY not set. "
                "Add it to your .env file: GROQ_API_KEY=your_key_here"
            )

        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "max_tokens": 150,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
            },
            timeout=10.0,
        )

        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text)

    def _extract_mood_history(self) -> list[str]:
        """
        Return mood history as a list of mood strings.
        """
        return []

    def reset(self):
        """Clear message history — call at start of a new session."""
        self._history = []


# ── Convenience function ──────────────────────────────────────────────────────

def detect_mood(text: str) -> MoodState:
    """One-shot mood detection without history tracking."""
    return MoodDetector().detect(text)


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = MoodDetector()

    test_inputs = [
        "I'm so excited, I just got a new job offer!!",
        "ugh I have 3 deadlines today and I haven't slept",
        "everything is fine I guess",
        "I can't stop thinking about what went wrong yesterday",
        "just got back from a run, feeling great!",
        "I don't know why but I just feel really empty today",
    ]

    print("\n🧠 Mood Detector — Test Run")
    print("=" * 55)

    for text in test_inputs:
        state = detector.detect(text)
        bar = "█" * int(state.mood_intensity * 15)
        print(f"\n  Input     : {text[:50]}")
        print(f"  Mood      : {state.current_mood:<10}  Intensity: {state.mood_intensity:.2f}  {bar}")

    print("\n" + "=" * 55)