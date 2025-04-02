"""Project settings."""

from pathlib import Path

# Instantiate and configure the project settings object
from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"


class ProjectHooks:
    @hook_impl
    def register_config_loader(self, conf_paths: list[str]) -> OmegaConfigLoader:
        return OmegaConfigLoader(conf_paths)

    @hook_impl
    def after_context_created(self, context):
        """Hook to execute after the Kedro context is created."""
        # Create necessary directories
        for dir_name in [
            "raw",
            "01_raw",
            "02_intermediate",
            "03_primary",
            "04_model",
            "05_model_input",
            "06_reporting",
            "plots",
        ]:
            (OUTPUT_DIR / dir_name).mkdir(parents=True, exist_ok=True)


# Define hooks to run before and after each pipeline node
class NodeHooks:
    @hook_impl
    def before_node_run(self, node, catalog, inputs):
        """Hook to execute before each node runs."""
        print(f"Running node: {node.name}")

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        """Hook to execute after each node runs."""
        print(f"Completed node: {node.name}")


HOOKS = (ProjectHooks(), NodeHooks())
