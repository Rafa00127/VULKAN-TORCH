"""`torch.nn.functional` 的对应物 —— Python 这侧的便利算子包装。

**这一层只给 Python。** C# 那边刻意贴着 C++ 的原始算子写法
（`Ops.Add(Ops.Linear(x, w), b)`），跟 `src/ops.h` 一一对应；对齐 torch 手感是
Python 侧的单独需求，所以方便方法都加在这里，不去动 `ops.cpp` / `c_api.cpp`。

每个包装都只是**把已有算子拼起来**，不新增算子、不重编：
`from vulkantorch.functional import *` 会覆盖掉 `_vulkantorch` 里同名的那个，
两参数时转发回原来的实现，所以调用点写几个参数都行。
"""

import vulkantorch._vulkantorch as _C

__all__ = ["linear"]


def linear(x, w, b=None):
    """`x @ wᵀ`，可选加 bias —— 对齐 `torch.nn.functional.linear(input, weight, bias=None)`。

    `w` 存成 PT `(out, in)`（和 `nn.Linear` 同向）。第二参数省略时就是
    `vt.linear(x, w)`，纯转发，一个节点都不多建。

    带 bias 时是 `add(linear(x, w), b)`：图上还是同样那两个节点，后端照样融成一次
    `MUL_MAT_ADD` dispatch，所以不比手写慢（见 CLAUDE.md 的那条附注）。
    bias 直接广播即可 —— 但这是 **linear** 的便利：conv 的 bias 得按通道轴 reshape
    （`ocr_vt/nn.py` 的 `_cadd` 就是干这个的），不在这一层。
    """
    y = _C.linear(x, w)
    return y if b is None else _C.add(y, b)
