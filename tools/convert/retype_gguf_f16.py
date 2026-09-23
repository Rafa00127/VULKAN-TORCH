"""Rewrite a GGUF's BF16 tensors as F16, in place at the byte level.

    python tools/convert/retype_gguf_f16.py in.gguf out.gguf

Every tensor keeps its byte count (BF16 and F16 are both 2 bytes per element) and GGUF
pads each tensor's data, so **all offsets and sizes stay valid** — the metadata block,
the tensor table and the data layout are copied verbatim, and only two things change:

* the type field of each BF16 tensor entry (BF16=30 -> F16=1), and
* that tensor's data, converted element-wise.

The conversion is exact where it matters: BF16's 7 mantissa bits fit inside F16's 10, so
the value only needs re-biasing (plus rounding when it lands in F16's subnormal range or
overflows its exponent range). Going through float32 does exactly that — bf16->f32 is
lossless, and numpy's f32->f16 rounds once, to nearest-even.

Why: on this backend the two dtypes take different matmul kernels (F16 x F32 gets the
f16_f32 pipeline, BF16 x F32 the dequant family), so the F16 copy is the A/B for
"is the BF16 path leaving performance on the table".
"""
import argparse
import os
import shutil
import struct

GGUF_MAGIC = b"GGUF"

# gguf_metadata_value_type
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)
_FIXED_FMT = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
              _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d"}

GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
ALIGNMENT = 32
CHUNK = 8 << 20


def _read_string(f):
    return f.read(struct.unpack("<Q", f.read(8))[0])


def _skip_value(f, vtype):
    """Advance past one metadata value; returns its size in bytes."""
    start = f.tell()
    fmt = _FIXED_FMT.get(vtype)
    if fmt is not None:
        f.seek(struct.calcsize(fmt), 1)
    elif vtype == _STRING:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == _ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _skip_value(f, et)
    else:
        raise ValueError(f"unknown GGUF metadata type {vtype}")
    return f.tell() - start


def retype(src, dst, from_type=GGML_TYPE_BF16, to_type=GGML_TYPE_F16):
    if from_type != GGML_TYPE_BF16 or to_type != GGML_TYPE_F16:
        raise SystemExit("only BF16 -> F16 is implemented (same element size)")
    shutil.copyfile(src, dst)
    n_conv = 0
    with open(dst, "r+b") as f:
        if f.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{dst}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version != 3:
            raise SystemExit(f"{dst}: unsupported GGUF version {version}")
        for _ in range(n_kv):
            _read_string(f)
            _skip_value(f, struct.unpack("<I", f.read(4))[0])

        # tensor table: name, n_dims, ne[], type(u32), offset(u64)
        entries = []
        for _ in range(n_tensors):
            name = _read_string(f).decode("utf-8", "replace")
            nd = struct.unpack("<I", f.read(4))[0]
            ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            type_pos = f.tell()
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            entries.append((name, ne, type_pos, ttype, off))

        data_start = f.tell()
        if data_start % ALIGNMENT:
            data_start += ALIGNMENT - (data_start % ALIGNMENT)

        import numpy as np
        for name, ne, type_pos, ttype, off in entries:
            if ttype != from_type:
                continue
            n_elems = 1
            for d in ne:
                n_elems *= d
            nbytes = n_elems * 2
            if n_elems == 0:
                continue
            pos = data_start + off
            todo = nbytes
            while todo:
                take = min(CHUNK, todo)
                f.seek(pos)
                raw = f.read(take)
                if len(raw) != take:
                    raise SystemExit(f"{dst}: truncated data for {name}")
                # bf16 -> f32 is exact (shift the 16 bits up), f32 -> f16 rounds once
                f32 = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
                with np.errstate(over="ignore"):
                    out = f32.astype(np.float16)
                f.seek(pos)
                f.write(out.tobytes())
                pos += take
                todo -= take
            f.seek(type_pos)
            f.write(struct.pack("<I", to_type))
            n_conv += 1
    return n_conv, n_tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    n_conv, n_tot = retype(a.src, a.dst)
    sa, sb = os.path.getsize(a.src), os.path.getsize(a.dst)
    print(f"{n_conv}/{n_tot} tensors retyped BF16 -> F16")
    print(f"size: {sa} -> {sb} bytes ({'same' if sa == sb else 'DIFFERENT!'})")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
