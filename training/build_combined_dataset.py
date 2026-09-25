from pathlib import Path
import shutil
import random

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dataset_combined"

# Final class IDs for the current proof-of-concept scope
CLASS_NAMES = [
    "key",         # 0
    "wallet",      # 1
    "bottle",      # 2
    "can",         # 3
    "book",        # 4
    "smartphone",  # 5
    "fork",        # 6
    "spoon",       # 7
    "pencil",      # 8
    "apple",       # 9
]

# Source datasets and class-ID remapping.
#
# None = ignore that source class.
#
# Keys:
#   0,1,2 -> key
#
# Bottle:
#   source 0 = bottle
#   source 1 = can, but we ignore it because we have
#   a dedicated Can dataset.
DATASETS = {
    "key": {
        "path": ROOT / "dataset_keys",
        "map": {0: 0, 1: 0, 2: 0},
    },

    "wallet": {
        "path": ROOT / "dataset_wallet",
        "map": {0: 1},
    },

    "bottle": {
        "path": ROOT / "dataset_bottle",
        "map": {0: 2, 1: None},
    },

    "can": {
        "path": ROOT / "dataset_can",
        "map": {0: 3},
    },

    "book": {
        "path": ROOT / "dataset_book",
        "map": {0: 4},
    },

    "smartphone": {
        "path": ROOT / "dataset_smartphone",
        "map": {0: 5},
    },

    "fork": {
        "path": ROOT / "dataset_fork",
        "map": {0: 6},
    },

    "spoon": {
        "path": ROOT / "dataset_spoon",
        "map": {0: 7},
    },

    "pencil": {
        "path": ROOT / "dataset_pencil",
        "map": {0: 8},
    },

    "apple": {
        "path": ROOT / "dataset_apple",
        "map": {0: 9},
    },
}

# Fixed seed means Bottle/Spoon get the same validation
# split every time this script is run.
RANDOM_SEED = 42
VALID_FRACTION = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_output():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "valid"):
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def get_images(image_dir):
    if not image_dir.exists():
        return []

    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def remap_label(source_label, destination_label, class_map):
    """
    Remap source YOLO class IDs into our combined class IDs.
    Returns True if at least one annotation was retained.
    """

    new_lines = []

    if source_label.exists():
        for line in source_label.read_text().splitlines():
            parts = line.split()

            if len(parts) < 5:
                continue

            source_class = int(parts[0])

            if source_class not in class_map:
                continue

            target_class = class_map[source_class]

            # Explicitly ignored source class
            if target_class is None:
                continue

            parts[0] = str(target_class)
            new_lines.append(" ".join(parts))

    destination_label.write_text("\n".join(new_lines))

    return len(new_lines) > 0


def copy_sample(dataset_name, dataset_path, image, source_split,
                target_split, class_map):

    source_label = (
        dataset_path
        / source_split
        / "labels"
        / f"{image.stem}.txt"
    )

    # Prefix filenames to prevent collisions between datasets.
    output_stem = f"{dataset_name}_{image.stem}"

    destination_image = (
        OUTPUT
        / target_split
        / "images"
        / f"{output_stem}{image.suffix.lower()}"
    )

    destination_label = (
        OUTPUT
        / target_split
        / "labels"
        / f"{output_stem}.txt"
    )

    # Create/remap label first.
    retained = remap_label(
        source_label,
        destination_label,
        class_map
    )

    # Keep the image even when no relevant annotation remains.
    # This gives YOLO legitimate background/negative examples.
    shutil.copy2(image, destination_image)

    return retained


def process_dataset(dataset_name, config):
    dataset_path = config["path"]
    class_map = config["map"]

    train_images = get_images(dataset_path / "train" / "images")
    valid_images = get_images(dataset_path / "valid" / "images")

    if not train_images:
        raise RuntimeError(
            f"No training images found for {dataset_name}: "
            f"{dataset_path}"
        )

    print(f"\n===== {dataset_name.upper()} =====")
    print(f"Source train images: {len(train_images)}")
    print(f"Source valid images: {len(valid_images)}")

    # If the source dataset already has validation data, preserve it.
    if valid_images:
        final_train = train_images
        final_valid = valid_images

        train_source_split = "train"
        valid_source_split = "valid"

    else:
        # Bottle and Spoon currently have no validation folders,
        # so create a deterministic 80/20 split from source train.
        rng = random.Random(RANDOM_SEED)

        shuffled = train_images.copy()
        rng.shuffle(shuffled)

        valid_count = max(
            1,
            round(len(shuffled) * VALID_FRACTION)
        )

        final_valid = shuffled[:valid_count]
        final_train = shuffled[valid_count:]

        train_source_split = "train"
        valid_source_split = "train"

        print(
            f"Created validation split: "
            f"{len(final_train)} train / "
            f"{len(final_valid)} valid"
        )

    retained_train = 0
    retained_valid = 0

    for image in final_train:
        if copy_sample(
            dataset_name,
            dataset_path,
            image,
            train_source_split,
            "train",
            class_map,
        ):
            retained_train += 1

    for image in final_valid:
        if copy_sample(
            dataset_name,
            dataset_path,
            image,
            valid_source_split,
            "valid",
            class_map,
        ):
            retained_valid += 1

    print(f"Final train images: {len(final_train)}")
    print(f"Final valid images: {len(final_valid)}")
    print(f"Train images with target annotations: {retained_train}")
    print(f"Valid images with target annotations: {retained_valid}")


def write_yaml():
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)

    yaml_text = (
        "train: train/images\n"
        "val: valid/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )

    (OUTPUT / "data.yaml").write_text(yaml_text)


def verify_labels():
    counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for split in ("train", "valid"):
        label_dir = OUTPUT / split / "labels"

        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text().splitlines():
                parts = line.split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id not in counts:
                    raise RuntimeError(
                        f"Unexpected class ID {class_id} "
                        f"in {label_path}"
                    )

                counts[class_id] += 1

    print("\n===== COMBINED ANNOTATION COUNTS =====")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_id}: {class_name:<12} "
            f"{counts[class_id]} annotations"
        )


def main():
    make_output()

    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config)

    write_yaml()
    verify_labels()

    train_count = len(
        get_images(OUTPUT / "train" / "images")
    )

    valid_count = len(
        get_images(OUTPUT / "valid" / "images")
    )

    print("\n===== COMPLETE =====")
    print(f"Combined training images: {train_count}")
    print(f"Combined validation images: {valid_count}")
    print(f"Dataset location: {OUTPUT}")


if __name__ == "__main__":
    main()