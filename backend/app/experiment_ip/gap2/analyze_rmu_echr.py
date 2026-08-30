from pathlib import Path
import json
from collections import Counter


DATASET_PATH = Path(
    "data/experiments/gap2/rmu_echr_annotations.json"
)


def main():

    print("=" * 70)
    print("RMU:ECHR GAP 2 — DATASET & LABEL ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    print(f"\nTotal records: {len(dataset)}")

    # ---------------------------------------------------------
    # BASIC COUNTS
    # ---------------------------------------------------------

    documents = set(record["document_id"] for record in dataset)

    actors = Counter(
        record["actor"]
        for record in dataset
    )

    argument_types = Counter(
        record["argument_type"]
        for record in dataset
    )

    print(f"Unique documents: {len(documents)}")
    print(f"Unique actors: {len(actors)}")
    print(f"Unique argument types: {len(argument_types)}")

    # ---------------------------------------------------------
    # ACTOR DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("ACTOR DISTRIBUTION")
    print("-" * 70)

    for actor, count in actors.most_common():

        percentage = (count / len(dataset)) * 100

        print(
            f"{actor:45} "
            f"{count:6} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # ARGUMENT TYPE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("ARGUMENT TYPE DISTRIBUTION")
    print("-" * 70)

    for arg_type, count in argument_types.most_common():

        percentage = (count / len(dataset)) * 100

        print(
            f"{arg_type:60} "
            f"{count:6} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # PASSAGE LENGTH
    # ---------------------------------------------------------

    lengths = [
        record["end"] - record["begin"]
        for record in dataset
    ]

    print("\n" + "-" * 70)
    print("PASSAGE LENGTH STATISTICS")
    print("-" * 70)

    print(f"Minimum characters:  {min(lengths)}")
    print(f"Maximum characters:  {max(lengths)}")
    print(f"Average characters:  {sum(lengths) / len(lengths):.2f}")

    # ---------------------------------------------------------
    # LENGTH BUCKETS
    # ---------------------------------------------------------

    buckets = {
        "< 100 chars": 0,
        "100–499 chars": 0,
        "500–999 chars": 0,
        "1000–1999 chars": 0,
        "2000–4999 chars": 0,
        "5000+ chars": 0,
    }

    for length in lengths:

        if length < 100:
            buckets["< 100 chars"] += 1

        elif length < 500:
            buckets["100–499 chars"] += 1

        elif length < 1000:
            buckets["500–999 chars"] += 1

        elif length < 2000:
            buckets["1000–1999 chars"] += 1

        elif length < 5000:
            buckets["2000–4999 chars"] += 1

        else:
            buckets["5000+ chars"] += 1

    print("\nLength distribution:")

    for bucket, count in buckets.items():

        percentage = (count / len(dataset)) * 100

        print(
            f"{bucket:20} "
            f"{count:6} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # RARE LABELS
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("RARE ARGUMENT TYPES")
    print("-" * 70)

    rare_labels = [
        (label, count)
        for label, count in argument_types.items()
        if count < 50
    ]

    for label, count in sorted(
        rare_labels,
        key=lambda x: x[1]
    ):
        print(f"{label:60} {count}")

    # ---------------------------------------------------------
    # EXAMPLES PER ARGUMENT TYPE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLE TEXT FOR EACH ARGUMENT TYPE")
    print("=" * 70)

    examples = {}

    for record in dataset:

        label = record["argument_type"]

        if label not in examples:
            examples[label] = record

    for label in argument_types:

        record = examples[label]

        text = " ".join(
            record["text"].split()
        )

        print("\n" + "-" * 70)
        print(f"LABEL: {label}")
        print(f"ACTOR: {record['actor']}")
        print(f"TEXT: {text[:500]}")

    # ---------------------------------------------------------
    # ACTOR × ARGUMENT TYPE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTOR × ARGUMENT TYPE")
    print("=" * 70)

    actor_argument = {}

    for record in dataset:

        actor = record["actor"]
        arg_type = record["argument_type"]

        if actor not in actor_argument:
            actor_argument[actor] = Counter()

        actor_argument[actor][arg_type] += 1

    for actor in sorted(actor_argument):

        print("\n" + "-" * 70)
        print(f"ACTOR: {actor}")

        for arg_type, count in actor_argument[actor].most_common():

            print(
                f"  {arg_type:55} {count}"
            )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()