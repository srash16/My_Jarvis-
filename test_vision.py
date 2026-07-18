#!/usr/bin/env python3
"""
Simple test script for the CameraWatcher component.
Tests the vision system in isolation: captures from webcam, sends to Gemini Vision,
and prints the description.

Usage:
    python test_vision.py
    Hold an object up to the webcam, press Ctrl+C to stop.
"""

import os
import signal
import sys
import time

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Import the CameraWatcher from vision.py (assumed to be in the same directory)
try:
    from vision import CameraWatcher
except ImportError:
    print("Error: Could not import CameraWatcher from vision.py")
    print("Make sure vision.py exists in the current directory and is importable.")
    sys.exit(1)

def main():
    # Get Google API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it.")
        sys.exit(1)

    # Create the genai client
    client = genai.Client(api_key=api_key)

    # Define a simple callback that just prints the description
    def description_callback(description: str):
        # Print with a timestamp for clarity
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Vision: {description}")

    # Create the camera watcher
    watcher = CameraWatcher(client, description_callback)

    # Set up signal handler for clean shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping watcher...")
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    # Also handle SIGTERM for good measure
    signal.signal(signal.SIGTERM, signal_handler)

    # Start watching
    print("Starting camera watcher...")
    watcher.start()
    print("Watching... hold an object up to the webcam, Ctrl+C to stop")

    try:
        # Keep the main thread alive until interrupted
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # This should be caught by the signal handler, but just in case
        print("\nStopping watcher...")
        watcher.stop()

if __name__ == "__main__":
    main()