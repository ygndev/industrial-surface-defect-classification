
import csv
import hashlib
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "splits.csv"

CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
SEED = 42


def get_label(path):
    prefix = path.stem.split("_")[0]
    label = prefix if prefix in CLASSES else path.parent.name

    if label not in CLASSES:
        raise ValueError(f"Unknown class: {path}")

    return label


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    paths = sorted(DATA_DIR.rglob("*.bmp"))

    if len(paths) != 1800:
        raise ValueError(f"Expected 1800 images, found {len(paths)}")

   
    unique_paths = []
    seen_hashes = {}

    for path in paths:
        digest = file_hash(path)
        label = get_label(path)

        if digest in seen_hashes:
            original = seen_hashes[digest]

            if get_label(original) != label:
                raise ValueError(
                    f"Conflicting labels: {original} and {path}"
                )

            print(
                f"Skipping duplicate: {path.name} "
                f"(same as {original.name})"
            )
            continue

        seen_hashes[digest] = path
        unique_paths.append(path)

    paths = unique_paths
    print(f"Unique images: {len(paths)}")


    labels = [get_label(path) for path in paths]

    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            paths,
            labels,
            train_size=0.70,
            random_state=SEED,
            stratify=labels,
        )
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_labels,
    )

    rows = []

    for split, split_paths, split_labels in (
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ):
        for path, label in zip(split_paths, split_labels):
            rows.append({
                "path": path.relative_to(ROOT).as_posix(),
                "label": label,
                "split": split,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["path", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["label"]) for row in rows)

    print("\nClass distribution:")
    for label in CLASSES:
        print(
            f"{label}: "
            f"train={counts['train', label]}, "
            f"val={counts['val', label]}, "
            f"test={counts['test', label]}"
        )

    print(f"\nSaved: {OUTPUT}")
    print(f"Total records: {len(rows)}")


if __name__ == "__main__":
    main()
