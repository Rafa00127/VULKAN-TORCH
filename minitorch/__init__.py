"""minitorch — a minimal PyTorch-style tensor library on ggml's Vulkan backend.

The compiled extension ``_minitorch`` ships inside this package and is
self-contained: ggml (base + CPU + Vulkan) is linked statically into it, so
``import minitorch`` needs nothing else beyond the Vulkan driver.
"""

from minitorch._minitorch import *  # noqa: F401,F403
