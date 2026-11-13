"""Output paths for analysis; input data under /data remains read-only."""
from pathlib import Path
from wrk_gen.queries import new_directory


def output_directory(path, inputs=()):
    destination=Path(path).expanduser().resolve()
    if destination.is_relative_to(Path('/data')):
        raise ValueError('Analysis output must be outside the read-only /data directory')
    return new_directory(destination,inputs)
