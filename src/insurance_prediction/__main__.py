"""Insurance Prediction file for ensuring the package is executable as `insurance-prediction` and `python -m insurance_prediction`."""

import importlib
from pathlib import Path

from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project


def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = next((_ for _ in plugins if _.name == "run"), None)
        if run:
            # use run command from installed plugin if it exists
            return run.load()
        # use run command from the framework project
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def main():
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run()


if __name__ == "__main__":
    main() 