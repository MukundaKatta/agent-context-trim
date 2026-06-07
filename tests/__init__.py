"""Test package for agent-context-trim.

Adds the ``src`` layout directory to ``sys.path`` so the test suite can be
run directly from a checkout with the standard library only::

    python3 -m unittest discover -s tests

without first installing the package (e.g. ``pip install -e .``).  When the
package *is* installed, this is a harmless no-op.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
