import os
import subprocess
import sys

def test_core_imports_and_cli():
    script="""
import sys
from sparql_wm_bench import catalog, workloads, policies, analysis, cli
assert not any(n in sys.modules for n in ('torch','dgl','jpype'))
cli.main(['--help'])
"""
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'workload' in result.stdout
