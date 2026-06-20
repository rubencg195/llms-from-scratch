"""Code injected at the top of every lab namespace before its cells run.

Makes notebook code safe to execute non-interactively and reproducibly:
  * force a headless matplotlib backend (no GUI windows / no blocking)
  * neutralise ``plt.show()`` so it never blocks a test run
  * seed Python / NumPy / Torch RNGs for repeatable results
  * silence noisy warnings

This is prepended, never written back to the lab file — the real lab code is
left untouched.
"""

PREAMBLE = r'''
import os as _os
import warnings as _warnings
_warnings.filterwarnings("ignore")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    _plt.show = lambda *a, **k: None  # never block the test run
except Exception:
    pass

import random as _random
_random.seed(0)
try:
    import numpy as _np
    _np.random.seed(0)
except Exception:
    pass
try:
    import torch as _torch
    _torch.manual_seed(0)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(0)
except Exception:
    pass
'''
