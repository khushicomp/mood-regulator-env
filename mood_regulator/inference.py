import subprocess
import sys
import os

# Install dependencies first
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "openai", "httpx", "python-dotenv", "pydantic", "fastapi", "uvicorn"])

# Force Python to see newly installed packages
import importlib
import site
importlib.reload(site)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run
from mood_regulator.baseline_agent import main

if __name__ == "__main__":
    main()