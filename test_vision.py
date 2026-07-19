#!/usr/bin/env python3
"""
Test on-demand vision: one capture, face-blur, Gemini describe, then exit.

Usage:
    python test_vision.py
    Hold an object up to the webcam when prompted.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

try:
    from vision import capture_and_describe_once
except ImportError:
    print("Error: Could not import capture_and_describe_once from vision.py")
    sys.exit(1)


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print("Hold something up to the camera — capturing in 2 seconds...")
    import time
    time.sleep(2)

    description = capture_and_describe_once(client)
    print(f"Vision: {description}")


if __name__ == "__main__":
    main()
