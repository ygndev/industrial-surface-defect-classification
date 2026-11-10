
import hashlib
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


groups = defaultdict(list)

for path in sorted(DATA_DIR.rglob("*.bmp")):
    groups[sha256(path)].append(path)

duplicate_groups = [
    paths for paths in groups.values()
    if len(paths) > 1
]

if not duplicate_groups:
    print("No exact duplicates found.")
else:
    for number, paths in enumerate(duplicate_groups, start=1):
        print(f"\nDuplicate group {number}:")

        for path in paths:
            print(f"  {path.relative_to(DATA_DIR)}")

    print(f"\nDuplicate groups: {len(duplicate_groups)}")
    print(
        "Extra duplicate files:",
        sum(len(paths) - 1 for paths in duplicate_groups)
    )
