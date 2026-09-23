"""vulkantorch — a minimal PyTorch-style tensor library on ggml's Vulkan backend.

The compiled extension ``_vulkantorch`` ships inside this package and is
self-contained: ggml (base + CPU + Vulkan) is linked statically into it, so
``import vulkantorch`` needs nothing else beyond the Vulkan driver.
"""

from vulkantorch._vulkantorch import *  # noqa: F401,F403
from vulkantorch.factory import *  # noqa: F401,F403
# after _vulkantorch: the torch-style wrappers here shadow the bare ops by design
from vulkantorch.functional import *  # noqa: F401,F403
import vulkantorch.methods  # noqa: F401  (patches Tensor/Graph in place; exports nothing)
