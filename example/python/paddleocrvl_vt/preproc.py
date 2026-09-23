"""Image preprocessing for PaddleOCR-VL.

Two independent resamplers live here, and they are *different algorithms* — do not
unify them:

* the input image is resized with ``mtmd-image.cpp``'s ``resize_bilinear``, which
  has align-corners geometry and quantizes each output pixel back to uint8;
* the learned position embedding is resized with ``ggml_interpolate`` in
  BILINEAR|ANTIALIAS mode, i.e. torch's antialiased triangle filter in float32.

Both are done here on the host, in numpy. That is not a shortcut: the Vulkan
backend in ``third_party/ggml`` has no ``GGML_OP_INTERPOLATE``, so the position
embedding would otherwise fall back to the CPU backend or fail.
"""
import numpy as np

PATCH = 14
MERGE = 2
ALIGN = PATCH * MERGE          # 28: the image grid is aligned to whole 2x2 merge blocks
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


# --- llama.cpp's round/floor/ceil "by factor" helpers (std::round semantics) ----------

def _round_by(x, f):
    # std::round: half away from zero. Python's round() is banker's rounding, so spell
    # it out rather than using it.
    return int(np.floor(x / f + 0.5)) * f


def _floor_by(x, f):
    return int(np.floor(x / f)) * f


def _ceil_by(x, f):
    return int(np.ceil(x / f)) * f


def smart_size(width, height, min_pixels, max_pixels):
    """``calc_size_preserved_ratio`` (transformers' "smart_resize"): aspect preserving,
    snapped to a multiple of ALIGN, then clamped into [min_pixels, max_pixels]."""
    w_bar = max(ALIGN, _round_by(width, ALIGN))
    h_bar = max(ALIGN, _round_by(height, ALIGN))

    if max_pixels > 0 and h_bar * w_bar > max_pixels:
        beta = np.sqrt(width * height / max_pixels)
        h_bar = max(ALIGN, _floor_by(height / beta, ALIGN))
        w_bar = max(ALIGN, _floor_by(width / beta, ALIGN))
    elif min_pixels > 0 and h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (width * height))
        h_bar = _ceil_by(height * beta, ALIGN)
        w_bar = _ceil_by(width * beta, ALIGN)
    return w_bar, h_bar


def resize_bilinear(img, target_w, target_h):
    """``img_tool::resize_bilinear``: float lerp, uint8 out, align-corners geometry."""
    src_h, src_w = img.shape[:2]
    x_ratio = (src_w - 1) / (target_w - 1) if target_w > 1 else 0.0
    y_ratio = (src_h - 1) / (target_h - 1) if target_h > 1 else 0.0

    px = np.arange(target_w, dtype=np.float32) * np.float32(x_ratio)
    py = np.arange(target_h, dtype=np.float32) * np.float32(y_ratio)
    x0 = np.minimum(px.astype(np.int64), src_w - 1)
    y0 = np.minimum(py.astype(np.int64), src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    xf = (px - x0).astype(np.float32)[None, :, None]
    yf = (py - y0).astype(np.float32)[:, None, None]

    src = img.astype(np.float32)
    p00 = src[y0[:, None], x0[None, :]]
    p10 = src[y0[:, None], x1[None, :]]
    p01 = src[y1[:, None], x0[None, :]]
    p11 = src[y1[:, None], x1[None, :]]
    # lerp(s, e, t) = s + (e - s) * t, in that order, then a down-cast (truncation).
    top = p00 + (p10 - p00) * xf
    bottom = p01 + (p11 - p01) * xf
    return (top + (bottom - top) * yf).astype(np.uint8)


def prepare_image(img, min_pixels, max_pixels):
    """PIL image -> uint8 HWC array at the resize target, matching the dyn_size
    preprocessor plus its PAD_CEIL padding (which is what `image_resize_pad`
    defaults to for this projector, and is not overridden per-model)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    src = np.asarray(img, dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    target_w, target_h = smart_size(src_w, src_h, min_pixels, max_pixels)

    if (target_w, target_h) == (src_w, src_h):
        return src

    scale = min(target_w / src_w, target_h / src_h)
    new_w = min(int(np.ceil(src_w * scale)), target_w)
    new_h = min(int(np.ceil(src_h * scale)), target_h)

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    out[off_y:off_y + new_h, off_x:off_x + new_w] = resize_bilinear(src, new_w, new_h)
    return out


def to_input_tensor(img_hwc):
    """HWC uint8 -> flat float32 bytes for the ViT's ``inp_raw`` (PT [1, 3, H, W];
    ggml ne = [W, H, C, N], so the channel is the slow axis and W the fast one),
    normalized by (x/255 - mean) / std."""
    h, w = img_hwc.shape[:2]
    chw = np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None], dtype=np.float32)
    chw /= 255.0
    chw -= IMAGE_MEAN
    chw /= IMAGE_STD
    return [1, 3, h, w], chw.tobytes()


def _triangle_filter(x):
    return np.maximum(1.0 - np.abs(x), 0.0).astype(np.float32)


def _antialias_weights(src_n, dst_n):
    """One axis of ggml's BILINEAR|ANTIALIAS upscale: a normalized triangle filter
    around each output sample (torch's ``F.interpolate(..., antialias=True)``)."""
    sf = dst_n / src_n
    support = max(1.0, 1.0 / sf)
    inv = 1.0 / support
    pos = (np.arange(dst_n, dtype=np.float32) + 0.5) / np.float32(sf)
    lo = np.maximum((pos - support + 0.5).astype(np.int64), 0)
    hi = np.minimum((pos + support + 0.5).astype(np.int64), src_n)

    w = np.zeros((dst_n, src_n), dtype=np.float32)
    for i in range(dst_n):
        sx = np.arange(lo[i], hi[i], dtype=np.float32)
        w[i, lo[i]:hi[i]] = _triangle_filter((sx - pos[i] + 0.5) * np.float32(inv))
    total = w.sum(axis=1, keepdims=True)
    np.divide(w, total, out=w, where=total > 0)
    return w


def resize_position_embd(pos_embd, n_w, n_h, n_per_side=27):
    """``clip_graph::resize_position_embeddings`` (bilinear + antialias).

    ``pos_embd`` is PT [n_per_side**2, n_embd] in patch order (row-major, x fastest);
    returns PT [n_w*n_h, n_embd] in the same order, resized from the square 27x27
    grid to the image's patch grid.

    The filter is separable and the per-output normalizer is the product of the two
    axis normalizers, so this is two normalized passes rather than the 2-D loop.
    """
    if n_w == n_per_side and n_h == n_per_side:
        return pos_embd
    n_embd = pos_embd.shape[1]

    # The grid is stored row-major over (y, x) with x fastest -- the same order the
    # ViT numbers its own patches in -- so reshape gives [y, x, e]. ggml's chain then
    # resizes the x axis to the target *width* and the y axis to the target height,
    # and re-flattens row-major, i.e. back to the ViT's patch order.
    grid = pos_embd.reshape(n_per_side, n_per_side, n_embd)                  # [y, x, e]
    tmp = np.einsum("jw,hwe->hje", _antialias_weights(n_per_side, n_w), grid)  # x -> n_w
    out = np.einsum("ih,hje->ije", _antialias_weights(n_per_side, n_h), tmp)   # y -> n_h
    return np.ascontiguousarray(out.reshape(n_h * n_w, n_embd))               # row = y*n_w + x
