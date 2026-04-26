#!/usr/bin/env python3
"""One-command setup for Neural Study OS"""
import os, sys, subprocess, platform
from pathlib import Path

def run(cmd): print(f"> {cmd}"); subprocess.run(cmd, shell=True)

def main():
    print("\n" + "="*50 + "\n  NEURAL STUDY OS - Setup Wizard\n" + "="*50)
    v = sys.version_info
    if v.major < 3 or v.minor < 9: print("ERROR: Python 3.9+ required"); sys.exit(1)
    print(f"Step 1: Python {v.major}.{v.minor}.{v.micro} OK")
    print("Step 2: Installing Python deps...")
    run("pip install -r requirements.txt")
    print("Step 3: Installing Node.js deps...")
    run("npm install")
    print("Step 4: Setting up Ollama...")
    sys = platform.system()
    if sys == "Windows": print("  Download from https://ollama.com/download/windows")
    elif sys == "Darwin": print("  Run: brew install ollama")
    else: print("  Run: curl -fsSL https://ollama.com/install.sh | sh")
    print("  Then: ollama pull llama3.2:3b")
    print("Step 5: Creating directories...")
    for d in ["data","data/uploads","data/chroma_db","data/videos","templates/manim"]:
        Path(d).mkdir(parents=True, exist_ok=True); print(f"  Created {d}")
    print("\n" + "="*50 + "\n  Setup Complete!\n  Run: python main.py\n  Open: http://localhost:5000\n" + "="*50)

if __name__ == "__main__": main()
