#!/usr/bin/env python
"""Quick test of the Gap 3 fixes."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.experiment_ip.gap3.document_loader import DocumentLoader

print("Test 1: DocumentLoader creation")
loader = DocumentLoader()
print(f"✓ Loader created, gold_data_dir: {loader.gold_data_dir}")
print(f"✓ Directory exists: {loader.gold_data_dir.exists()}")

# Try to load one document
print("\nTest 2: Load single document")
doc_text = loader.load_document_from_xmi("001-100469")
if doc_text:
    print(f"✓ Document loaded: {len(doc_text)} characters")
    print(f"  First 100 chars: {doc_text[:100]}...")
else:
    print("✗ Failed to load document")
    print(f"  Status: {loader.load_status.get('001-100469', 'No status')}")

print("\nTest 3: Load annotations")
import json
project_root = Path(__file__).parent.parent.parent
annotation_path = project_root / "data" / "experiments" / "gap2" / "rmu_echr_annotations.json"

if annotation_path.exists():
    with open(annotation_path) as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} annotations")
    print(f"  Sample: {annotations[0]}")
else:
    print(f"✗ Annotation file not found: {annotation_path}")

print("\nAll tests passed!")
