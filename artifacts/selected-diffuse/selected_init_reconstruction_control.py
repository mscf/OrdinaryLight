"""Identical-shader negative control for stochastic reconstruction parity."""
import sys
from pathlib import Path
import types
scripts=Path.cwd()/"components/OrdinaryLight/scripts"
sys.path.insert(0,str(scripts))
path=scripts/"selected_diffuse_render_experiment.py"
source=path.read_text()
branch="        if fixed_mask is not None:\n            return pipeline\n"
assert source.count(branch)==1
source=source.replace(branch,"")
module=types.ModuleType("selected_diffuse_render_experiment")
module.__file__=str(path)
sys.modules[module.__name__]=module
exec(compile(source,str(path),"exec"),module.__dict__)
from profile_primary_prefixes import main
main()
