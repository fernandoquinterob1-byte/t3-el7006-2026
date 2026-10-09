"""Import helpers for the source distribution of Neural Process Family.

The upstream project is a research repository rather than a regular PyPI
package.  The checked-out source lives in ``third_party/Neural-Process-Family``
and is added to ``sys.path`` only when the project imports ``npf``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NPF_ROOT = REPO_ROOT / "third_party" / "Neural-Process-Family"


def ensure_npf_importable() -> None:
    """Make the vendored upstream source importable, failing clearly if absent."""

    if str(NPF_ROOT) not in sys.path:
        sys.path.insert(0, str(NPF_ROOT))
    if not (NPF_ROOT / "npf" / "__init__.py").is_file():
        raise ImportError(
            "Neural Process Family source is missing. Clone "
            "https://github.com/YannDubs/Neural-Process-Family into "
            f"{NPF_ROOT}."
        )


ensure_npf_importable()

