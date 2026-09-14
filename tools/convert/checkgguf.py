"""List a GGUF's tensors: names, ggml type (F32/F16/Q4_0/...), and shape.

    python tools/convert/checkgguf.py model.gguf [--all] [--type F16] [--prefix blk.]

Shows the type histogram (F32/F16/quant mix), then either the first 60 tensors or
(--all) every one. `ne` is the ggml order (fastest dimension first); the PT column
is the reversed PyTorch shape. Handy for checking what a converter actually wrote,
or which tensors are still F32 (norms/biases) vs F16.
"""
import struct
import sys
import collections

# enum ggml_type (ggml.h) -> display name
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    if t in (0, 1, 7): f.read(1)
    elif t in (2, 3): f.read(2)
    elif t in (4, 5, 6): f.read(4)
    elif t == 8: read_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        for _ in range(cnt): skip_val(f, et)
    elif t in (10, 11, 12): f.read(8)
    else: raise ValueError(f"bad type {t}")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    path = args[0]
    show_all = "--all" in args or "-a" in args
    type_filter = prefix = None
    for i, a in enumerate(args):
        if a == "--type": type_filter = args[i + 1]
        if a == "--prefix": prefix = args[i + 1]

    tensors = []
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_str(f)
            skip_val(f, struct.unpack("<I", f.read(4))[0])
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            tid = struct.unpack("<I", f.read(4))[0]
            f.read(8)  # offset
            tensors.append((name, ne, tid))

    hist = collections.Counter(TYPE_NAMES.get(t, f"#{t}") for _, _, t in tensors)
    print(f"{path}")
    print(f"  version={ver} tensors={n_tensors} kv={n_kv}")
    print(f"  types: " + "  ".join(f"{k}={v}" for k, v in hist.most_common()))

    sel = tensors
    if type_filter:
        sel = [t for t in sel if TYPE_NAMES.get(t[2], f"#{t[2]}").lower() == type_filter.lower()]
    if prefix:
        sel = [t for t in sel if t[0].startswith(prefix)]
    print(f"  showing {len(sel) if show_all else min(60, len(sel))}/{len(sel)} tensors:")
    n = len(sel) if show_all else min(60, len(sel))
    print(f"    {'name':52s} {'type':>7}  {'ne (ggml, fastest first)':>26}  PT (reversed)")
    for name, ne, tid in sel[:n]:
        tn = TYPE_NAMES.get(tid, f"#{tid}")
        ne_s = "[" + ",".join(str(x) for x in ne) + "]"
        pt_s = "[" + ",".join(str(x) for x in reversed(ne)) + "]"
        print(f"    {name:52s} {tn:>7}  {ne_s:>26}  {pt_s}")


if __name__ == "__main__":
    main()
