"""Call higgs_tts.dll's encode_ref via ctypes and compare codes with ref.npz.

Run with the main env python (needs the model gguf + the TTS build-vk DLLs).
"""
import ctypes
import os
import sys

import numpy as np

RELEASE = r"<local>/external\HiggsTTSOnline\native\build-vk\bin\Release"
GGUF = r"<local>/reader-app\models\higgs-v3-tts-q8_0.gguf"
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
REF_NPZ = os.path.join(REFS, "ref.npz")
N_CB = 8

os.add_dll_directory(RELEASE)
dll = ctypes.CDLL(os.path.join(RELEASE, "higgs_tts.dll"))

dll.higgs_tts_load.restype = ctypes.c_void_p
dll.higgs_tts_load.argtypes = [ctypes.c_char_p]
dll.higgs_tts_encode_ref.restype = ctypes.c_int
dll.higgs_tts_encode_ref.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
dll.higgs_tts_free.argtypes = [ctypes.c_void_p]

print("loading model:", GGUF)
h = dll.higgs_tts_load(GGUF.encode())
if not h:
    print("higgs_tts_load failed")
    sys.exit(1)

ref = np.load(REF_NPZ)
codes_ref = ref["codes"]  # [T, 8]


def load_wav(path):
    import wave

    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a


wav = load_wav(os.path.join(_ROOT, "data", "ref_audio", "melinaref_24k.wav"))  # 24k mono
print("wav", wav.shape, "codes_ref", codes_ref.shape)

max_frames = 4000
out = (ctypes.c_int32 * (max_frames * N_CB))()
n = dll.higgs_tts_encode_ref(h, wav.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             len(wav), out)
print("dll encode_ref returned T =", n)
if n > 0:
    codes_dll = np.ctypeslib.as_array(out)[: n * N_CB].reshape(n, N_CB)
    print("dll codes", codes_dll.shape, "first row", codes_dll[0].tolist())
    if codes_dll.shape == codes_ref.shape:
        mism = int((codes_dll != codes_ref).sum())
        print(f"mismatch: {mism}/{codes_dll.size}")
    else:
        print("shape differs -> different T")
        m = min(n, codes_ref.shape[0])
        mism = int((codes_dll[:m] != codes_ref[:m]).sum())
        print(f"head mismatch: {mism}/{m * N_CB}")
dll.higgs_tts_free(h)
