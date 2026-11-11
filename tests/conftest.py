"""Keep pytest temporary files and optional-library caches inside the project."""
import os
from pathlib import Path
ROOT=Path.cwd().resolve()
for key,leaf in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('DGLDEFAULTDIR','dgl'),('MPLCONFIGDIR','matplotlib')]:
    p=ROOT/'.local'/leaf; p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)

def pytest_configure(config):
    if config.option.basetemp is None: config.option.basetemp=str(ROOT/'.local'/'pytest-tmp')
