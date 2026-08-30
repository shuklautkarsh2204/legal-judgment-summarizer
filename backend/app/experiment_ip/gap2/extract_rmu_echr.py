from pathlib import Path
import xml.etree.ElementTree as ET
import json


# CHANGE THIS to your RMU:ECHR gold_data folder
GOLD_DATA_DIR = Path(
    r"D:\Projects\mining-legal-arguments\gold_data"
)

# Where our extracted dataset will be saved
OUTPUT_PATH = Path(
    "data/experiments/gap2/rmu_echr_annotations.json"
)


def extract_document_text(root):
    """
    Extract the original document text from the XMI CAS sofa.
    """
    for element in root.iter():
        if element.tag.endswith("Sofa"):
            text = element.attrib.get("sofaString")

            if text is not None:
                return text

    return None


def extract_annotations(root):
    """
    Extract LegalArgumentation annotations.
    """
    annotations = []

    for element in root.iter():

        if not element.tag.endswith("LegalArgumentation"):
            continue

        try:
            begin = int(element.attrib["begin"])
            end = int(element.attrib["end"])

            actor = element.attrib.get("Akteur")
            argument_type = element.attrib.get("ArgType")

            annotations.append({
                "begin": begin,
                "end": end,
                "actor": actor,
                "argument_type": argument_type
            })

        except (KeyError, ValueError):
            continue

    return annotations


def main():

    print("=" * 70)
    print("RMU:ECHR GAP 2 — XMI TO CLEAN DATASET")
    print("=" * 70)

    xmi_files = sorted(GOLD_DATA_DIR.glob("*.xmi"))

    print(f"\nXMI files found: {len(xmi_files)}")

    if not xmi_files:
        print("\nERROR: No XMI files found.")
        print(f"Check path:\n{GOLD_DATA_DIR}")
        return

    dataset = []

    documents_processed = 0
    total_annotations = 0
    invalid_spans = 0
    missing_text = 0

    for xmi_file in xmi_files:

        try:
            tree = ET.parse(xmi_file)
            root = tree.getroot()

            document_text = extract_document_text(root)

            if document_text is None:
                missing_text += 1
                print(f"WARNING: No document text found: {xmi_file.name}")
                continue

            annotations = extract_annotations(root)

            documents_processed += 1
            total_annotations += len(annotations)

            for annotation in annotations:

                begin = annotation["begin"]
                end = annotation["end"]

                # Validate character boundaries
                if (
                    begin < 0
                    or end > len(document_text)
                    or begin >= end
                ):
                    invalid_spans += 1
                    continue

                text = document_text[begin:end]

                dataset.append({
                    "document_id": xmi_file.stem,
                    "text": text,
                    "begin": begin,
                    "end": end,
                    "actor": annotation["actor"],
                    "argument_type": annotation["argument_type"]
                })

        except ET.ParseError as e:
            print(f"ERROR parsing {xmi_file.name}: {e}")

        except Exception as e:
            print(f"ERROR processing {xmi_file.name}: {e}")

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save clean dataset
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("EXTRACTION STATISTICS")
    print("=" * 70)

    print(f"\nDocuments processed:       {documents_processed}")
    print(f"Total legal annotations:   {total_annotations}")
    print(f"Clean dataset records:     {len(dataset)}")
    print(f"Invalid spans:             {invalid_spans}")
    print(f"Missing document text:     {missing_text}")

    print("\n" + "-" * 70)
    print("SAMPLE RECORDS")
    print("-" * 70)

    for record in dataset[:5]:

        print("\nDocument:", record["document_id"])
        print("Text:", record["text"][:300].replace("\n", " "))
        print("Begin:", record["begin"])
        print("End:", record["end"])
        print("Actor:", record["actor"])
        print("Argument type:", record["argument_type"])

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"\nDataset saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()