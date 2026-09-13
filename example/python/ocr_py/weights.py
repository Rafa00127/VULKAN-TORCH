"""Load a GGUF's tensors as device-resident vulkantorch tensors."""
import vulkantorch as mt


class Weights:
    def __init__(self, path, device, arena_bytes=1 << 30):
        self.file = mt.GgufFile(path, device, arena_bytes)
        self.t = {n: self.file.tensor(n) for n in self.file.names()}

    def __getitem__(self, name):
        return self.t[name]

    def __contains__(self, name):
        return name in self.t

    def __len__(self):
        return len(self.t)
