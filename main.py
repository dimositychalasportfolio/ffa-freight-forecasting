"""Runnable entry point for the FFA vessel-and-period forecasting experiment."""

import importlib.util
import os
import subprocess
import sys


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "openpyxl": "openpyxl",
}


def ensure_dependencies() -> None:
    """Install missing packages before importing the modelling module."""
    missing = [
        package
        for package, module in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        environment = {**os.environ, "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-color", "--progress-bar", "off", *missing],
            env=environment,
        )


def main() -> None:
    ensure_dependencies()
    from ffa_panel import main as run_panel

    run_panel()


if __name__ == "__main__":
    main()
