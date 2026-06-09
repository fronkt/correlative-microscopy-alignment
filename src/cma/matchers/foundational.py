"""Import-guarded shells for foundational dense matchers still being wired up.

`MatchAnythingMatcher` is now real (cma.matchers.matchanything). The
remaining shells (RoMaMatcher, ELoFTRMatcher) raise `MatcherNotInstalled`
at construction time until their weights are vendored.

When you're ready to wire one up:
    1. `pip install -e .[torch]`
    2. Vendor the upstream backbone source under `vendor/<name>/`
    3. Drop the model weights under `checkpoints/<name>/`
    4. Replace the `raise MatcherNotInstalled(...)` block with the real
       model construction + a `match` implementation that converts the
       dense flow output into a `Correspondences` object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cma.matchers.base import Correspondences, Matcher


class MatcherNotInstalled(RuntimeError):
    pass


def _need(msg: str) -> None:
    raise MatcherNotInstalled(msg)


# RoMaMatcher has been implemented (see cma.matchers.roma).
# Importing it from this module is still supported for backward compatibility.


class ELoFTRMatcher(Matcher):
    """Efficient LoFTR — semi-dense transformer-based matcher."""

    name = "eloftr"

    def __init__(
        self,
        weights_path: str | Path = "checkpoints/eloftr/eloftr_outdoor.ckpt",
        device: str = "cuda",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        _need(
            "ELoFTRMatcher is a shell. Install torch+kornia ("
            "`pip install -e .[torch]`), vendor ELoFTR, and provide weights at "
            f"{self.weights_path}."
        )

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:  # pragma: no cover
        raise MatcherNotInstalled("ELoFTRMatcher backend not implemented yet.")


# MatchAnythingMatcher has been implemented (see cma.matchers.matchanything).
# Importing it from this module is still supported for backward compatibility.
