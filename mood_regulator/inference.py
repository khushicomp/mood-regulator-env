import subprocess
import sys

# Auto-install required packages
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "openai", "httpx", "python-dotenv", "pydantic", "fastapi"
])

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from mood_regulator.baseline_agent import main

if __name__ == "__main__":
    main()