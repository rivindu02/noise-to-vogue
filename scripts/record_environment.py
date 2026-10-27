"""Record relevant dependency versions without exporting unrelated site packages."""
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import torch


def main():
    packages = {}
    for name in ("torch", "numpy", "Pillow", "fastapi", "uvicorn", "pydantic", "starlette", "httpx", "pytest", "setuptools"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    path = Path(__file__).resolve().parent.parent / "results/environment.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
