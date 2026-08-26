# Ordinary Light workbench showcase scripts

The Ordinary Light workbench discovers ordinary Python files in this directory
and in directories listed by `ORDINARYLIGHT_SHOWCASE_PATH`.

A minimal extension is:

```python
import ordinarylight as ol
from ordinarylight.integrations.workbench import Showcase

def build():
    scene = ol.Scene()
    # Add resources...
    return scene

SHOWCASE = Showcase("my-scene", "My scene", build)
```

Builders run lazily when selected rather than during application startup. A
script may define `SHOWCASES` with several entries. Camera presentation and
renderer defaults are optional metadata; renderer features remain expressed
through the normal Ordinary Light API.
