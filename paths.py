import os
from pathlib import Path

# use this path object to specify location
# of anything within repository
repo_root = Path(__file__).resolve().parent
repo_root_placeholder = "REPO_ROOT"

def evaluate_path(relative_path):
    """Replace the placeholder with the actual repo root path."""
    evaluated_path = relative_path
    if relative_path.startswith(repo_root_placeholder + "/"):
        evaluated_path = str(repo_root / relative_path[len(repo_root_placeholder)+1:])
    return evaluated_path
    