"""Read the GGUF key/value header.

``vulkantorch.GgufFile`` maps tensor *data* into device memory but exposes no
metadata, and this port needs two things out of the header that are not tensors:
the vision hparams (patch size, merge factor, image_mean/std) and the whole
sentencepiece vocabulary (103424 entries) for the tokenizer.

**Everything comes back as bytes.** llama.cpp stores token text as bytes and
matches/merges on bytes; a handful of entries (the ``<0xNN>`` byte-fallback
tokens, and the raw bytes sentencepiece uses for control characters) are not
valid UTF-8, so decoding them to str would silently corrupt the vocabulary.
Bytes in, bytes out; only the user-facing decode step at the end decodes.
"""
import struct

_MAGIC = b"GGUF"

# enum gguf_metadata_value_type
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, \
    _UINT64, _INT64, _FLOAT64 = range(13)

_FIXED = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}


def _read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def _read_value(f, vtype):
    fmt = _FIXED.get(vtype)
    if fmt is not None:
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(n)]
    raise ValueError(f"unknown GGUF metadata value type {vtype}")


def read_metadata(path):
    """Return ``{key: value}`` for the file's KV block.

    String values are bytes; arrays keep their element type (so
    ``tokenizer.ggml.tokens`` is a ``list[bytes]`` and
    ``paddleocr.rope.dimension_sections`` a ``list[int]``).
    """
    with open(path, "rb") as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version != 3:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        meta = {}
        for _ in range(n_kv):
            key = _read_string(f).decode("utf-8")
            meta[key] = _read_value(f, struct.unpack("<I", f.read(4))[0])
        return meta
