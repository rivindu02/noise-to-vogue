import gzip
import hashlib
from io import BytesIO
import struct
import pytest
import torch
from noise_to_vogue import data


def fixture_files(root, monkeypatch, bad_header=False):
    images = gzip.compress(struct.pack(">IIII", 1 if bad_header else 2051, 2, 28, 28) + bytes([0] * 784 + [255] * 784))
    labels = gzip.compress(struct.pack(">II", 2049, 2) + bytes([3, 9]))
    checksums = {}
    for name, content in [("t10k-images-idx3-ubyte.gz", images), ("t10k-labels-idx1-ubyte.gz", labels)]:
        (root / name).write_bytes(content)
        checksums[name] = hashlib.md5(content).hexdigest()
    monkeypatch.setattr(data, "FILES", checksums)


def test_idx_parser_normalizes_without_using_labels_as_inputs(tmp_path, monkeypatch):
    fixture_files(tmp_path, monkeypatch)
    x, y = data.load_data(tmp_path, train=False)
    assert x.shape == (2, 1, 28, 28)
    assert torch.all(x[0] == -1) and torch.all(x[1] == 1)
    assert y.tolist() == [3, 9]


def test_idx_header_is_validated(tmp_path, monkeypatch):
    fixture_files(tmp_path, monkeypatch, bad_header=True)
    with pytest.raises(ValueError, match="header"):
        data.load_data(tmp_path, train=False)


def test_corrupted_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: BytesIO(b"corrupt download"))
    with pytest.raises(ValueError, match="Checksum"):
        data.load_data(tmp_path, train=False)
    assert not list(tmp_path.iterdir())
