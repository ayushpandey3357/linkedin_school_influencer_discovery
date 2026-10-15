"""
Run script for EDXSO School Influencer Discovery Tool.
Launches the Streamlit dashboard on the default local port.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app.py"
    port = os.environ.get("PORT", "8501")
    print("=" * 65)
    print("  EDXSO School Influencer Discovery Tool")
    print("  Launching Streamlit Dashboard...")
    print(f"  Target File: {app_path}")
    print(f"  Port: {port}")
    print("=" * 65)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Failed to launch application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
