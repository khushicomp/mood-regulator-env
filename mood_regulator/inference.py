import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mood_regulator.baseline_agent import main

if __name__ == "__main__":
    main()