"""
Run script for EDXSO School Influencer Discovery Tool.
Launches the Streamlit dashboard on the default local port.
"""

import sys
import subprocess
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent / "app.py"
    print("=" * 65)
    print("  EDXSO School Influencer Discovery Tool")
    print("  Launching Streamlit Dashboard...")
    print(f"  Target File: {app_path}")
    print("=" * 65)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Failed to launch application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
