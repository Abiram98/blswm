"""Keep generated artifacts in the working tree and away from input data."""
from pathlib import Path


def output_path(path):
    root = Path.cwd().resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path == root or path.is_relative_to(root / 'data'):
        raise ValueError('Outputs must stay inside the working directory, outside data/')
    return path


def new_directory(path):
    path = output_path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
