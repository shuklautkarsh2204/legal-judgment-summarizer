from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


# CHANGE THIS PATH to your cloned RMU:ECHR gold_data folder
GOLD_DATA_DIR = Path(
    r"D:\Projects\mining-legal-arguments\gold_data"
)


def inspect_xmi_file(xmi_path: Path):
    tree = ET.parse(xmi_path)
    root = tree.getroot()

    # Namespace used by the custom legal annotation
    custom_ns = "http:///webanno/custom.ecore"
    legal_argument_tag = f"{{{custom_ns}}}LegalArgumentation"

    annotations = []

    for element in root.iter(legal_argument_tag):
        annotations.append({
            "begin": int(element.attrib["begin"]),
            "end": int(element.attrib["end"]),
            "actor": element.attrib.get("Akteur"),
            "arg_type": element.attrib.get("ArgType"),
        })

    return annotations


def main():

    xmi_files = list(GOLD_DATA_DIR.glob("*.xmi"))

    print("=" * 70)
    print("RMU:ECHR GAP 2 — ANNOTATION INSPECTION")
    print("=" * 70)

    print(f"\nXMI files found: {len(xmi_files)}")

    if not xmi_files:
        print("\nERROR: No .xmi files found.")
        print(f"Check this path:\n{GOLD_DATA_DIR}")
        return

    total_annotations = 0
    actor_counter = Counter()
    arg_type_counter = Counter()

    example_annotations = []

    for xmi_file in xmi_files:

        try:
            annotations = inspect_xmi_file(xmi_file)

            total_annotations += len(annotations)

            for annotation in annotations:
                actor_counter[annotation["actor"]] += 1
                arg_type_counter[annotation["arg_type"]] += 1

            # Keep examples from the first file that contains annotations
            if annotations and len(example_annotations) < 10:
                for annotation in annotations[:10]:
                    example_annotations.append(
                        (xmi_file.name, annotation)
                    )

        except Exception as e:
            print(f"\nERROR processing {xmi_file.name}:")
            print(e)

    print("\n" + "=" * 70)
    print("ANNOTATION STATISTICS")
    print("=" * 70)

    print(f"\nTotal documents:       {len(xmi_files)}")
    print(f"Total legal spans:     {total_annotations}")

    print("\n" + "-" * 70)
    print("ACTOR DISTRIBUTION")
    print("-" * 70)

    for actor, count in actor_counter.most_common():
        print(f"{actor:45} {count}")

    print("\n" + "-" * 70)
    print("ARGUMENT TYPE DISTRIBUTION")
    print("-" * 70)

    for arg_type, count in arg_type_counter.most_common():
        print(f"{arg_type:60} {count}")

    print("\n" + "=" * 70)
    print("ANNOTATION EXAMPLES")
    print("=" * 70)

    for filename, annotation in example_annotations:
        print(f"\nFile: {filename}")
        print(f"  Begin:     {annotation['begin']}")
        print(f"  End:       {annotation['end']}")
        print(f"  Actor:     {annotation['actor']}")
        print(f"  ArgType:   {annotation['arg_type']}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()