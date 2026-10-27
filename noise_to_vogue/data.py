"""Download checksum-verified original IDX files; no torchvision dependency."""
import gzip
import hashlib
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen
import numpy as np
import torch

BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
LABELS = ["T-shirt / top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_data(root="data", train=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if train else "t10k"
    arrays = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        path = root / name
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
            with urlopen(BASE + name, timeout=90) as response:
                content = response.read()
            if hashlib.md5(content).hexdigest() != FILES[name]:
                raise ValueError(f"Checksum mismatch: {name}")
            with tempfile.NamedTemporaryFile(dir=root, suffix=".download", delete=False) as handle:
                handle.write(content)
                tmp = Path(handle.name)
            try:
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        content = gzip.decompress(path.read_bytes())
        if "images" in kind:
            magic, n, h, w = struct.unpack(">IIII", content[:16])
            if (magic, h, w) != (2051, 28, 28) or len(content) != 16 + n * h * w:
                raise ValueError("Invalid image IDX header")
            arrays.append(torch.from_numpy(np.frombuffer(content[16:], dtype=np.uint8).copy()).reshape(n, 1, h, w).float() / 127.5 - 1)
        else:
            magic, n = struct.unpack(">II", content[:8])
            if magic != 2049 or len(content) != 8 + n:
                raise ValueError("Invalid label IDX header")
            arrays.append(torch.from_numpy(np.frombuffer(content[8:], dtype=np.uint8).copy()).long())
    if len(arrays[0]) != len(arrays[1]) or not ((arrays[1] >= 0) & (arrays[1] < 10)).all():
        raise ValueError("Invalid label count or values")
    return tuple(arrays)
