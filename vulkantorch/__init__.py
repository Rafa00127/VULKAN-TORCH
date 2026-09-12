"""vulkantorch — a minimal PyTorch-style tensor library on ggml's Vulkan backend.

The compiled extension ``_vulkantorch`` ships inside this package and is
self-contained: ggml (base + CPU + Vulkan) is linked statically into it, so
``import vulkantorch`` needs nothing else beyond the Vulkan driver.
"""

from vulkantorch._vulkantorch import *  # noqa: F401,F403
