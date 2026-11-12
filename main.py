"""Run the FFA forecasting experiment: python main.py

Install the dependencies first with: pip install -r requirements.txt
"""

import importlib.util
import sys


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
}


def check_dependencies() -> None:
    """Stop with a clear message if a package is missing, instead of installing it silently."""
    if missing := [
        package for module, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None
    ]:
        sys.exit(f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt")


def main() -> None:
    check_dependencies()
    from ffa_panel import main as run_panel

    run_panel()


if __name__ == "__main__":
    main()
