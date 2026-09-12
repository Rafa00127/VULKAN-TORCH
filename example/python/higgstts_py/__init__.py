from higgstts_py import _paths

_paths.setup()  # repo root on sys.path (vulkantorch) + native DLL dirs

__all__ = ["HiggsTTS"]


def __getattr__(name):
    # lazy: importing higgstts_py only bootstraps paths; heavy deps (model/tts)
    # load on first attribute access.
    if name == "HiggsTTS":
        from higgstts_py.tts import HiggsTTS
        return HiggsTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
