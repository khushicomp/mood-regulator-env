import subprocess
import sys
import os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "openai", "httpx", "python-dotenv", "pydantic", "fastapi", "uvicorn"])

# Set working directory and path
workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, workspace)

# Set required env vars if not present
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
if not os.environ.get("OPENAI_MODEL"):
    os.environ["OPENAI_MODEL"] = "llama-3.1-8b-instant"

# Run as module to preserve relative imports
import runpy
runpy.run_module("mood_regulator.baseline_agent", run_name="__main__", alter_sys=True)