#!/usr/bin/env python
"""Entry point script for running the insurance prediction project."""

import sys
from pathlib import Path
from typing import Any, Dict

from kedro.framework.hooks import _create_hook_manager
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def run_pipeline(pipeline_name: str = None, **kwargs: Dict[str, Any]):
    """Run the specified pipeline with the given parameters.

    Args:
        pipeline_name: Name of the pipeline to run. If None, the default pipeline will be run.
        **kwargs: Additional parameters to pass to the run command.
    """
    # Get the current project path
    project_path = Path.cwd()

    # Bootstrap the Kedro project
    metadata = bootstrap_project(project_path)

    # Create a Kedro session
    with KedroSession.create(
        metadata.package_name,
        project_path,
        hook_manager=_create_hook_manager(),
    ) as session:
        # Run the pipeline
        if pipeline_name:
            session.run(pipeline_name=pipeline_name, **kwargs)
        else:
            session.run(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run a specific pipeline if provided
        run_pipeline(sys.argv[1])
    else:
        # Run the default pipeline
        run_pipeline()
