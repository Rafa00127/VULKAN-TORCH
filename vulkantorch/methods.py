"""torch-style methods on ``Tensor`` and ``Graph``, monkeypatched onto the pybind types.

Nothing here needs the extension rebuilt: these are plain assignments to the *type
objects* (``Tensor.__add__ = ...``, ``Tensor.T = property(...)``). That works without
``py::dynamic_attr()`` -- that flag is only needed to set attributes on individual
*instances*, which this module never does.

The consequence is the constraint: every method below has to be expressible with
what the extension already binds. The two that are not -- ``Tensor.dtype`` and
``Tensor.is_contiguous`` -- are bound in C++ instead (``_vulkantorch.cpp``).

``__getitem__`` is deliberately absent: slicing returns a ggml *view*, and whether
that view aliases (and can be silently overwritten by later nodes reusing the same
storage) is the open question at the top of CLAUDE.md. Settle that first.
"""

import numpy as np

import vulkantorch._vulkantorch as _C
from vulkantorch.factory import _NP_OF, _resolve, to_numpy

__all__: list = []  # this module only patches; it exports no names of its own


def _tensor(x):
    return isinstance(x, _C.Tensor)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
# Scalar operands go through scale/scale_bias rather than a materialized 1-element
# tensor: the latter would need a Memory, which cannot be allocated inside a capture.

def _add(self, other):
    return _C.add(self, other) if _tensor(other) else _C.scale_bias(self, 1.0, float(other))


def _radd(self, other):
    return _C.scale_bias(self, 1.0, float(other))


def _sub(self, other):
    return _C.sub(self, other) if _tensor(other) else _C.scale_bias(self, 1.0, -float(other))


def _rsub(self, other):
    return _C.scale_bias(self, -1.0, float(other))


def _mul(self, other):
    return _C.mul(self, other) if _tensor(other) else _C.scale(self, float(other))


def _truediv(self, other):
    if _tensor(other):
        return _C.div(self, other)
    return _C.scale(self, 1.0 / float(other))


def _matmul(self, other):
    return _C.matmul(self, other)


def _neg(self):
    return _C.scale(self, -1.0)


# `__rtruediv__` (scalar / tensor) is intentionally missing: it needs a reciprocal,
# and the library has no such op. Python then raises a plain TypeError, which is the
# honest answer. `__rsub__` is fine -- scale_bias(-1, s) is exactly s - a.


# ---------------------------------------------------------------------------
# shape / layout
# ---------------------------------------------------------------------------

def _pt_perm(rank, indices):
    """Reported PT dims -> ``permute_pt``'s axes.

    ``permute_pt`` indexes the *4-dim padded* PyTorch order, but ``shape()`` reports the
    collapsed one: a rank-2 tensor lives at ggml ``ne = [cols, rows, 1, 1]``, so its
    reported dims 0 and 1 are padded axes 2 and 3. Feeding 0/1 silently permutes two
    size-1 axes and returns the input unchanged.
    """
    offset = 4 - rank
    perm = [0, 1, 2, 3]
    for k, j in enumerate(indices):
        perm[offset + k] = offset + j
    return perm


def _T(self):
    """torch's ``.T``: reverse all dims (for rank <= 2 that is the usual transpose)."""
    rank = len(self.shape)
    if rank > 4:
        raise ValueError(f"Tensor.T: rank {rank} unsupported (max 4)")
    return _C.permute_pt(self, *_pt_perm(rank, reversed(range(rank))))


def _mT(self):
    """torch's ``.mT``: swap the last two dims only."""
    return _transpose(self, -2, -1)


def _transpose(self, dim0, dim1):
    rank = len(self.shape)
    d0 = dim0 + rank if dim0 < 0 else dim0
    d1 = dim1 + rank if dim1 < 0 else dim1
    if not (0 <= d0 < rank and 0 <= d1 < rank):
        raise IndexError(f"Tensor.transpose: dims ({dim0}, {dim1}) out of range for rank {rank}")
    order = list(range(rank))
    order[d0], order[d1] = order[d1], order[d0]
    return _C.permute_pt(self, *_pt_perm(rank, order))


def _reshape(self, *shape):
    """``reshape`` with torch's ``-1`` inference; 1..4 dims (the library's limit)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dims = [int(d) for d in shape]
    if dims.count(-1) > 1:
        raise ValueError(f"Tensor.reshape: only one -1 allowed, got {dims}")
    if -1 in dims:
        known = 1
        for d in dims:
            if d != -1:
                known *= d
        if known <= 0 or self.numel % known:
            raise ValueError(
                f"Tensor.reshape: cannot infer -1 dividing {self.numel} elements into {dims}")
        dims[dims.index(-1)] = self.numel // known
    if not 1 <= len(dims) <= 4:
        raise ValueError(f"Tensor.reshape: only 1..4 dims supported, got {len(dims)}")
    return _C.reshape(self, dims)


def _flatten(self):
    return self.reshape(-1)


def _unsqueeze(self, dim):
    """Insert a size-1 dim.

    Note the library's ``.shape`` drops *leading* size-1 dims (it goes through
    ``ggml_n_dims``), so ``x.unsqueeze(0).shape == x.shape`` even though the
    underlying tensor really is rank+1 and downstream ops see it that way.
    """
    dims = list(self.shape)
    dims.insert(dim, 1)
    return self.reshape(*dims)


def _contiguous(self):
    return _C.contiguous(self)


def _cast(self, dtype):
    """``cast`` to another dtype. Also the target of ``.to()``."""
    if isinstance(dtype, _C.Device):
        raise NotImplementedError(
            "Tensor.to(device): moving between devices needs a host round-trip. Build the "
            "tensor on the target device instead (vt.zeros(..., device=dev)).")
    return _C.cast(self, _resolve(dtype))


# ---------------------------------------------------------------------------
# reading data
# ---------------------------------------------------------------------------

def _numpy(self, dtype=None):
    """Host copy as a numpy array. The dtype comes from the tensor itself."""
    if dtype is None:
        if self.dtype not in _NP_OF:
            raise TypeError(
                f"Tensor.numpy: no numpy equivalent for ggml type {self.dtype}; pass dtype= "
                f"explicitly.")
        dtype = _NP_OF[self.dtype]
    return to_numpy(self, dtype)


def _tolist(self):
    return self.numpy().tolist()


def _item(self):
    if self.numel != 1:
        raise ValueError(f"Tensor.item: expected 1 element, got {self.numel}")
    return self.numpy().reshape(-1)[0].item()


def _array(self, dtype=None, copy=None):
    array = self.numpy()
    return array if dtype is None else array.astype(dtype)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _graph_enter(self):
    self.enter()
    return self


def _graph_exit(self, exc_type, exc, tb):
    self.exit()
    return False  # never swallow the exception


def _graph_output(self, *tensors):
    """Mark tensors as outputs so their buffers are not reused. Call before exit()."""
    for t in tensors:
        t.mark_output()
    return tensors[0] if len(tensors) == 1 else tensors


# ---------------------------------------------------------------------------

for _name, _fn in {
    "__add__": _add, "__radd__": _radd, "__sub__": _sub, "__rsub__": _rsub,
    "__mul__": _mul, "__rmul__": _mul, "__truediv__": _truediv,
    "__matmul__": _matmul, "__neg__": _neg,
}.items():
    setattr(_C.Tensor, _name, _fn)

for _name, _member in {
    "T": _T,
    "mT": _mT,
    "transpose": _transpose,
    "reshape": _reshape,
    "flatten": _flatten,
    "unsqueeze": _unsqueeze,
    "contiguous": _contiguous,
    "cast": _cast,
    "to": _cast,
    "numpy": _numpy,
    "tolist": _tolist,
    "item": _item,
    "__array__": _array,
}.items():
    setattr(_C.Tensor, _name, property(_member) if _name in ("T", "mT") else _member)

setattr(_C.Graph, "__enter__", _graph_enter)
setattr(_C.Graph, "__exit__", _graph_exit)
setattr(_C.Graph, "output", _graph_output)
