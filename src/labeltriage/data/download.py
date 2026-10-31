"""Checksum-verified fetching of every raw file the project uses.

Files land under ``<data_dir>/raw/<relpath>``. A file that is already present and matches its
checksum is never downloaded again. Files marked ``manual`` (ChaosNLI's Dropbox zip) cannot be
fetched by script; :func:`fetch` explains where to put them instead.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawFile:
    relpath: str
    url: str
    checksum: str  # "<algorithm>:<hex digest>"
    manual: bool = False


CIFAR10N_BASE = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main"
CIFAR10_1_BASE = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets"

FILES: dict[str, RawFile] = {
    "cifar10h": RawFile(
        "cifar10h/cifar10h-raw.zip",
        "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip",
        "sha256:4b0283f9b2ae72a616eeb4a2a38474cc2a238eb041bfb8df5ead3a8a47f35aa3",
    ),
    "cifar10n_labels": RawFile(
        "cifar10n/CIFAR-10_human.pt",
        f"{CIFAR10N_BASE}/data/CIFAR-10_human.pt",
        "sha256:873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310",
    ),
    "cifar10n_side_info": RawFile(
        "cifar10n/side_info_cifar10N.csv",
        f"{CIFAR10N_BASE}/side_info_cifar10N.csv",
        "sha256:50fcbe619a003d0b804a20ece51447dcdb48978c60e1202371d08302b2e14351",
    ),
    "varierr": RawFile(
        "varierr/varierr.json",
        "https://raw.githubusercontent.com/mainlp/VariErr-NLI/main/varierr.json",
        "sha256:c8f9a634a69d72a6c390047ac242562b80a8caac08db19714db3f98bd40e13ea",
    ),
    "chaosnli": RawFile(
        "chaosnli/chaosNLI_v1.0.zip",
        "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=0",
        "",  # recorded the first time the official zip is placed by hand
        manual=True,
    ),
    "chaosnli_mnli_mirror": RawFile(
        "chaosnli/chaos_mnli.jsonl",
        "https://huggingface.co/datasets/tasksource/chaos-mnli-ambiguity/resolve/main/chaos_mnli.jsonl",
        "sha256:07d756705106c5aeab5216326deff7a16992430a466e19d6214065e73efcb9b5",
    ),
    "cifar10_1_data": RawFile(
        "cifar10_1/cifar10.1_v6_data.npy",
        f"{CIFAR10_1_BASE}/cifar10.1_v6_data.npy",
        "sha256:2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6",
    ),
    "cifar10_1_labels": RawFile(
        "cifar10_1/cifar10.1_v6_labels.npy",
        f"{CIFAR10_1_BASE}/cifar10.1_v6_labels.npy",
        "sha256:ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448",
    ),
    "cifar10_c": RawFile(
        "cifar10_c/CIFAR-10-C.tar",
        "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar",
        "md5:56bf5dcef84df0e2308c6dcbcbbd8499",
    ),
}


class ChecksumError(RuntimeError):
    pass


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str | Path, checksum: str) -> bool:
    """True if ``path`` matches ``checksum``. An empty checksum always passes."""
    if not checksum:
        return True
    algorithm, expected = checksum.split(":", 1)
    return file_digest(path, algorithm) == expected


def fetch(name: str, data_dir: str | Path) -> Path:
    """Return the local path of raw file ``name``, downloading and verifying it if needed."""
    spec = FILES[name]
    path = Path(data_dir) / "raw" / spec.relpath

    if path.exists():
        if not verify(path, spec.checksum):
            raise ChecksumError(f"{path} does not match {spec.checksum}; delete it and refetch")
        return path

    if spec.manual:
        raise FileNotFoundError(
            f"{name} must be downloaded by hand from {spec.url} and saved as {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(spec.url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    if not verify(tmp, spec.checksum):
        tmp.unlink()
        raise ChecksumError(f"download of {name} from {spec.url} failed its checksum")
    tmp.replace(path)
    return path
