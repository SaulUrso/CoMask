import importlib.abc
import importlib.util
import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))

# Maps fedml's internal module names to local override files.
# Packages (those with __init__.py) carry a submodule_search_locations list so
# that relative imports inside the override files resolve correctly.
_OVERRIDES = {
    "fedml.simulation.sp.fedavg": (
        os.path.join(_BASE, "fedavg", "__init__.py"),
        [os.path.join(_BASE, "fedavg")],
    ),
    "fedml.simulation.sp.fedavg.client": (
        os.path.join(_BASE, "fedavg", "client.py"),
        None,
    ),
    "fedml.simulation.sp.fedavg.fedavg_api": (
        os.path.join(_BASE, "fedavg", "fedavg_api.py"),
        None,
    ),
    "fedml.simulation.sp.scaffold": (
        os.path.join(_BASE, "scaffold", "__init__.py"),
        [os.path.join(_BASE, "scaffold")],
    ),
    "fedml.simulation.sp.scaffold.client": (
        os.path.join(_BASE, "scaffold", "client.py"),
        None,
    ),
    "fedml.simulation.sp.scaffold.scaffold_trainer": (
        os.path.join(_BASE, "scaffold", "scaffold_trainer.py"),
        None,
    ),
}


class _OverrideFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _OVERRIDES:
            return None
        file_path, search_locations = _OVERRIDES[fullname]
        if not os.path.isfile(file_path):
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            file_path,
            submodule_search_locations=search_locations,
        )


sys.meta_path.insert(0, _OverrideFinder())
