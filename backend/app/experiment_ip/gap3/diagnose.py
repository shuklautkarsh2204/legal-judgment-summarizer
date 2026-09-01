#!/usr/bin/env python
"""Quick diagnostic for Gap 3 experiment."""

import sys
from pathlib import Path

print("Starting Gap 3 diagnostic...", flush=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    print("Attempting imports...", flush=True)
    from backend.app.services.passage_builder import PassageBuilder
    print("✓ PassageBuilder imported", flush=True)
    
    from backend.app.services.semantic_analyzer import SemanticAnalyzer
    print("✓ SemanticAnalyzer imported", flush=True)
    
    print("\nAttempting to instantiate SemanticAnalyzer...", flush=True)
    sa = SemanticAnalyzer()
    print("✓ SemanticAnalyzer instantiated", flush=True)
    
    print("\nAll diagnostics passed!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
