"""Upload a built AmbiNoise release to the Hugging Face Hub. Dry run unless --yes is given.

    python scripts/upload_ambinoise.py release/ambinoise --repo-id <user>/ambinoise        # dry run
    python scripts/upload_ambinoise.py release/ambinoise --repo-id <user>/ambinoise --yes  # upload

The dataset repository is created private (pass --public to change that). The script refuses to
upload while the card's licence is still `other`, because the owner must first check CIFAR-10H's
and ChaosNLI's terms. It needs `huggingface_hub` and a token from `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def card_licence(release: Path) -> str | None:
    text = (release / "README.md").read_text(encoding="utf8")
    m = re.search(r"^license:\s*(\S+)\s*$", text.split("---", 2)[1], flags=re.M)
    return m.group(1) if m else None


def plan(release: Path) -> list[tuple[str, int]]:
    return sorted(
        (str(p.relative_to(release)), p.stat().st_size) for p in release.rglob("*") if p.is_file()
    )


def main(argv: list[str] | None = None, api=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--yes", action="store_true", help="actually upload (default: dry run)")
    args = parser.parse_args(argv)

    if not (args.release / "README.md").exists():
        print(f"{args.release} has no README.md; build it with scripts/release_ambinoise.py")
        return 1
    files = plan(args.release)
    licence = card_licence(args.release)
    print(
        f"{len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB -> {args.repo_id} "
        f"({'public' if args.public else 'private'}), licence: {licence}"
    )
    for name, size in files[:20]:
        print(f"  {name}  ({size / 1e3:.0f} kB)")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")

    if licence in (None, "other"):
        print("REFUSING: set the card's licence after checking CIFAR-10H's and ChaosNLI's terms.")
        return 2
    if not args.yes:
        print("Dry run only. Re-run with --yes to upload.")
        return 0

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(args.release), repo_id=args.repo_id, repo_type="dataset")
    print("uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
