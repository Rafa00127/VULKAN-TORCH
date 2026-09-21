"""torch-style tensor factories.

Every factory here builds a **Memory leaf**: a device-resident tensor that owns
its storage and outlives any graph -- the closest thing this library has to a
torch leaf (``nn.Parameter``). The other way to get a tensor, ``Graph.input()``,
is a graph *input*: no storage of its own, allocated by the scheduler at compute
time and copied in from the CPU. Use that one for activations inside a capture.

Consequence: these factories **cannot be called while a graph capture is open**.
``Memory`` refuses to allocate then, so call them before ``g.enter()``.

The device comes from a module-level default (``set_default_device``); if none
was ever set, a ``Runtime`` is created lazily and its GPU is used, so
``vt.tensor([[1., 2.], [3., 4.]])`` works with no setup.
"""

import numpy as np

import vulkantorch._vulkantorch as _C

__all__ = [
    "F32", "F16", "I32",
    "default_runtime", "set_default_device", "default_device",
    "tensor", "zeros", "ones", "full", "eye", "empty", "rand", "randn",
    "from_numpy", "as_tensor", "to_numpy",
    "zeros_like", "ones_like", "full_like",
]

# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------

# ggml_type values (third_party/ggml/include/ggml.h): F32 = 0, F16 = 1, I32 = 26.
# Only these three are exposed -- the Vulkan backend has no op path for I8 (24),
# I16 (25), I64 (27) or F64 (28), so they could be allocated as raw bytes but
# nothing could ever compute on them.
F32 = _C.F32
F16 = _C.F16
I32 = 26

_NP_OF = {
    F32: np.dtype(np.float32),
    F16: np.dtype(np.float16),
    I32: np.dtype(np.int32),
}

_INT_DATA = (
    "integer data has no default dtype: torch would infer int64 here, and this "
    "library has no I64 path on Vulkan. Pass dtype=vt.I32 explicitly (or "
    "dtype='float32' if you meant values, not indices)."
)


def _bad_dtype(dtype):
    return (
        f"unsupported dtype {dtype!r}: only float32, float16 and int32 exist here. "
        "The Vulkan backend has no op path for I8/I16/I64/F64, so those are not "
        "exposed."
    )


def _resolve(dtype):
    """dtype= accepts a ggml type int (``vt.I32``), a numpy dtype, or a name string."""
    if isinstance(dtype, bool):
        raise TypeError(_bad_dtype(dtype))
    if isinstance(dtype, int):
        if dtype not in _NP_OF:
            raise TypeError(_bad_dtype(dtype))
        return dtype
    try:
        wanted = np.dtype(dtype)
    except TypeError:
        raise TypeError(_bad_dtype(dtype)) from None
    for ggml_type, numpy_dtype in _NP_OF.items():
        if numpy_dtype == wanted:
            return ggml_type
    raise TypeError(_bad_dtype(dtype))


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

_default_rt = None
_default_dev = None


def default_runtime():
    """The ``Runtime`` the default device lives on, created on first use.

    Held at module scope for the process lifetime -- it *owns* the ggml backend,
    so anything pointing at it (a Device, a Graph, a Memory) is only valid while
    it is alive.
    """
    global _default_rt
    if _default_rt is None:
        _default_rt = _C.Runtime()
    return _default_rt


def set_default_device(device, runtime=None):
    """Send the factories to ``device`` from now on.

    ``Device`` holds a raw ``ggml_backend_t``; the ``Runtime`` owns it. So pass
    ``runtime=`` whenever ``device`` came from a Runtime you own --
    ``set_default_device(rt.cpu(), runtime=rt)`` -- otherwise a Runtime that goes
    out of scope leaves the default device dangling.
    """
    global _default_dev, _default_rt
    if runtime is not None:
        _default_rt = runtime
    elif _default_rt is None:
        default_runtime()  # make sure the process keeps one backend alive
    _default_dev = device


def default_device():
    """The device the factories use: whatever was set, else ``default_runtime().gpu()``."""
    if _default_dev is not None:
        return _default_dev
    return default_runtime().gpu()


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _where(device):
    return default_device() if device is None else device


def _as_shape(args):
    """zeros(2, 3), zeros((2, 3)) and zeros([2, 3]) all mean shape (2, 3)."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        dims = [int(d) for d in args[0]]
    else:
        dims = [int(d) for d in args]
    if not dims:
        raise ValueError(
            "empty shape: this library has no 0-d tensor (ggml needs at least 1 dim)")
    return dims


def _leaf(array, ggml_type, device):
    """Materialize ``array`` as a device-resident leaf, on its very own Memory."""
    array = np.ascontiguousarray(array, _NP_OF[ggml_type])
    memory = _C.Memory(_where(device), array.nbytes)
    # The returned Tensor keeps `memory` alive (src/memory.cpp), so letting the
    # Python handle go out of scope here is safe.
    return memory.tensor(list(array.shape), ggml_type, array.tobytes())


def _defaulted(dtype):
    return F32 if dtype is None else _resolve(dtype)


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------

def tensor(data, dtype=None, device=None):
    """Leaf from a nested sequence or a numpy array, like ``torch.tensor``.

    dtype is inferred from the data: floats -> F32, and **integers raise** rather
    than picking a type, because torch would say int64 and there is no I64 here.
    """
    array = np.asarray(data)
    if dtype is None:
        if array.dtype.kind in "iub":
            raise TypeError(_INT_DATA)
        return _leaf(array, F32, device)  # float64 -> F32, as in torch
    ggml_type = _resolve(dtype)
    return _leaf(array.astype(_NP_OF[ggml_type]), ggml_type, device)


def zeros(*shape, dtype=None, device=None):
    ggml_type = _defaulted(dtype)
    return _leaf(np.zeros(_as_shape(shape), _NP_OF[ggml_type]), ggml_type, device)


def ones(*shape, dtype=None, device=None):
    ggml_type = _defaulted(dtype)
    return _leaf(np.ones(_as_shape(shape), _NP_OF[ggml_type]), ggml_type, device)


def full(shape, fill_value, dtype=None, device=None):
    """Constant-filled leaf. An integer fill needs an explicit dtype (same rule as tensor)."""
    if dtype is None and isinstance(fill_value, (int, np.integer)):
        raise TypeError(_INT_DATA)
    ggml_type = _defaulted(dtype)
    return _leaf(np.full(_as_shape((shape,)), fill_value, _NP_OF[ggml_type]), ggml_type, device)


def eye(n, m=None, dtype=None, device=None):
    ggml_type = _defaulted(dtype)
    return _leaf(np.eye(n, m, dtype=_NP_OF[ggml_type]), ggml_type, device)


def empty(*shape, dtype=None, device=None):
    """Uninitialized leaf -- skips the host->device upload entirely.

    Goes through ``Memory.tensor(shape, dtype, b"")``: the binding still hands a
    (non-null) pointer down, but ``ggml_backend_tensor_set`` returns early on a
    zero size, so nothing is written.
    """
    ggml_type = _defaulted(dtype)
    dims = _as_shape(shape)
    nbytes = int(np.prod(dims)) * _NP_OF[ggml_type].itemsize
    return _C.Memory(_where(device), nbytes).tensor(dims, ggml_type, b"")


def rand(*shape, dtype=None, device=None):
    """Uniform [0, 1), F32 by default (like ``torch.rand``)."""
    ggml_type = _defaulted(dtype)
    return _leaf(np.random.rand(*_as_shape(shape)), ggml_type, device)


def randn(*shape, dtype=None, device=None):
    """Standard normal, F32 by default (like ``torch.randn``)."""
    ggml_type = _defaulted(dtype)
    return _leaf(np.random.randn(*_as_shape(shape)), ggml_type, device)


# ---------------------------------------------------------------------------
# numpy bridge
# ---------------------------------------------------------------------------

def from_numpy(array, device=None):
    """Leaf from a numpy array, **keeping its dtype** -- so float64/int64/bool raise.

    This is the faithful bridge; use ``tensor()`` for torch-style inference.
    """
    array = np.asarray(array)
    try:
        ggml_type = _resolve(array.dtype)
    except TypeError:
        raise TypeError(
            f"from_numpy: numpy dtype {array.dtype} has no vulkantorch equivalent. "
            f"Cast first, e.g. array.astype(np.float32) "
            f"(or np.int32 for index tensors).") from None
    return _leaf(array, ggml_type, device)


def as_tensor(array, dtype=None, device=None):
    """Like ``from_numpy``, but ``dtype=`` casts the values first."""
    if dtype is None:
        return from_numpy(array, device=device)
    ggml_type = _resolve(dtype)
    return _leaf(np.asarray(array).astype(_NP_OF[ggml_type]), ggml_type, device)


def to_numpy(t, dtype=None):
    """Read a tensor back to a numpy array (a copy -- the data lives on the device).

    The dtype comes from the tensor unless you pass one. ``Tensor.numpy()`` is the
    same thing as a method.
    """
    if dtype is None:
        if t.dtype not in _NP_OF:
            raise TypeError(
                f"to_numpy: no numpy equivalent for ggml type {t.dtype}; pass dtype= explicitly.")
        dtype = _NP_OF[t.dtype]
    return np.frombuffer(t.to_bytes(), dtype).reshape(list(t.shape)).copy()


# ---------------------------------------------------------------------------
# *_like
# ---------------------------------------------------------------------------

def zeros_like(x, dtype=None, device=None):
    """Shape and dtype from ``x`` (dtype= overrides)."""
    return zeros(x.shape, dtype=x.dtype if dtype is None else dtype, device=device)


def ones_like(x, dtype=None, device=None):
    return ones(x.shape, dtype=x.dtype if dtype is None else dtype, device=device)


def full_like(x, fill_value, dtype=None, device=None):
    return full(x.shape, fill_value, dtype=x.dtype if dtype is None else dtype, device=device)
