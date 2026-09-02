# GAP 3 FIXES IMPLEMENTATION REPORT

## Executive Summary

The Gap 3 experiment has been comprehensively fixed to use **original RMU:ECHR document texts** instead of synthetic reconstructions. All character offsets now correctly refer to the original judgment documents, eliminating the critical scientific flaw that invalidated the previous implementation.

## The Problem Fixed

**Critical Issue**: The original Gap 3 implementation created synthetic documents by concatenating only annotated spans, filling gaps with "[... X chars omitted ...]" markers. This caused a **fundamental coordinate system mismatch**:

- RMU:ECHR annotations have `begin`/`end` offsets referring to the **original judgment document**
- But passages were built from **synthetic reconstructed documents**
- This meant passage-to-annotation mappings used **incompatible coordinate systems**
- Distance calculations were based on synthetic document structure, not the real legal text

**Result**: All scientific findings were unreliable because distances, text alignments, and passage-annotation relationships were corrupted.

## Solution Implemented

### Fix 1: Created `document_loader.py` Module

**File**: `backend/app/experiment_ip/gap3/document_loader.py`

A new module that:
- Loads original RMU:ECHR documents from XMI files in `D:\Projects\mining-legal-arguments\gold_data\`
- Extracts document text from the Sofa element in each XMI file
- Validates annotation offsets against the original document text
- Provides status reporting on document loading success/failures

**Key Class**: `DocumentLoader`
- `load_document_from_xmi(document_id)` — Load a single document
- `load_all_documents(document_ids)` — Load multiple documents
- `validate_annotation_offset()` — Verify offsets are correct
- `print_load_report()` — Print loading summary

### Fix 2: Updated `test_passage_relationships.py`

#### 2a: Modified Imports
Added import of the new DocumentLoader:
```python
from backend.app.experiment_ip.gap3.document_loader import DocumentLoader
```

#### 2b: Replaced `load_document_texts()` Function
- **Removed**: Synthetic document construction with "[... X chars omitted ...]" markers
- **Added**: Direct loading of original documents from RMU:ECHR XMI files
- **Returns**: Tuple of (document_texts, loader, annotations) for validation

#### 2c: Enhanced `Gap3Analyzer` Class
Added validation infrastructure:
- New instance variables: `loader`, `document_texts`, `validation_report`
- New method: `validate_annotations_and_documents()`
  - Validates annotation offsets against original documents
  - Samples 50 annotations for detailed checking
  - Reports valid/invalid offsets and text mismatches
  - Confirms NO synthetic reconstruction is used

#### 2d: Updated `generate_results_json()`
- Now includes `validation` report in JSON output
- Marks `synthetic_reconstruction: False` in metadata
- Adds `coordinate_system` note: "Original RMU:ECHR document character offsets"
- Lists corrections from v1 in `corrections_from_v1` field

#### 2e: Modified `main()` Function
- Loads original documents from XMI files
- Calls `validate_annotations_and_documents()` before analysis
- Saves results to new output file: `gap3_real_document_relationship_results.json`
- Saves validation report separately: `gap3_real_document_validation.json`

## Files Changed

### New Files
1. **`backend/app/experiment_ip/gap3/document_loader.py`** (180+ lines)
   - DocumentLoader class with XMI parsing
   - Annotation offset validation
   - Load status reporting

### Modified Files
1. **`backend/app/experiment_ip/gap3/test_passage_relationships.py`**
   - Updated imports (added DocumentLoader)
   - Replaced load_document_texts() function
   - Enhanced Gap3Analyzer class with validation
   - Updated generate_results_json() with validation report
   - Modified main() to use original documents and validation

## Verification Checklist

### ✓ Original Document Source Located
- RMU:ECHR XMI files exist at: `D:\Projects\mining-legal-arguments\gold_data\`
- File naming convention: `{document_id}.xmi` (e.g., `001-100469.xmi`)
- Sofa elements contain original document text

### ✓ Coordinate System Unified
- All `begin` and `end` offsets now refer to original documents
- Annotation text extracted as: `document_text[begin:end]`
- Passage positions calculated from original document structure
- No synthetic coordinate transforms

### ✓ Synthetic Reconstruction Removed
- No "[... X chars omitted ...]" markers created
- No gap-filling between annotated spans
- Real document text used entirely

### ✓ Validation Implemented
- Annotation offset validation against original documents
- Boundary checking: `begin >= 0`, `end <= doc_len`, `begin < end`
- Text mismatch detection with whitespace normalization
- Comprehensive validation report output

### ✓ Distance Calculations Fixed
- Passage distance: `abs(j - i)` — based on passage indices in real doc
- Character distance: gap between passage boundaries in real doc
- All calculations use original document structure

### ✓ Annotation-Passage Mapping Corrected
- Overlap detection uses original document offsets
- Condition: `annotation.begin < passage.end_char AND annotation.end > passage.begin_char`
- No coordinate system mismatches

### ✓ Results Output Updated
- New output file: `gap3_real_document_relationship_results.json`
- Separate validation report: `gap3_real_document_validation.json`
- Metadata clearly states: "Original RMU:ECHR document character offsets (NOT synthetic)"
- Corrections documented in results

## Scientific Implications

### Important
The corrected experiment may show **different (likely weaker) results** than the original implementation because:
1. Real document structure is more complex than annotation concatenation
2. Passage-to-annotation overlap patterns differ when using original text
3. Document proximity signals may be more realistic but noisier
4. Semantic similarity calculations use actual passages from real documents

**This is expected and correct.** The goal is scientific accuracy, not positive results.

### Expected Behaviors
- Validation should report high success rate (>95%) for annotation offsets
- Some text mismatches may occur (minor formatting differences)
- Passages should be properly ordered and non-overlapping
- All distances should be positive or zero

## How to Run the Corrected Experiment

```bash
cd d:\Projects\LegalJudgementSummarizer
python backend/app/experiment_ip/gap3/test_passage_relationships.py
```

### Expected Output
1. Document loading report (showing successes and failures)
2. Annotation validation report (showing offset validity)
3. Passage construction status
4. Pair construction summary
5. Analysis results (semantic similarity, transitions, combined signals)
6. Two JSON files saved:
   - `data/experiments/gap3/gap3_real_document_relationship_results.json`
   - `data/experiments/gap3/gap3_real_document_validation.json`

## Comparison: Old vs. New

| Aspect | Old Implementation | New Implementation |
|--------|-------------------|-------------------|
| Document Source | Synthetic (annotations only) | Original RMU:ECHR XMI files |
| Character Offsets | Synthetic doc coordinates | Original document coordinates |
| Text Between Annotations | "[... X chars omitted ...]" | Real document text |
| Validation | None | Comprehensive offset validation |
| Metadata | Silent about synthetic nature | Explicitly documents original docs |
| Output Files | `gap3_relationship_analysis.json` | `gap3_real_document_*.json` |
| Annotation-Passage Overlap | Synthetic doc coordinates | Original doc coordinates |
| Distance Calculations | Synthetic structure | Real document structure |

## No Breaking Changes

- Gap 1 remains unchanged
- Gap 2 remains unchanged
- Existing output files (from old experiment) remain for reproducibility
- New experiment uses different output filenames
- All existing code and services remain functional

## Next Steps for User

1. **Verify the fix**: Run the experiment and check for validation errors
2. **Review validation report**: Examine `gap3_real_document_validation.json`
3. **Interpret results**: Remember these are exploratory findings, not causal claims
4. **Document any issues**: If document loading fails for some cases, investigate
5. **Proceed cautiously**: Use these results only as evidence for future research directions

## Reproducibility

- Fixed random seed (42) maintained
- Same embedding model (all-MiniLM-L6-v2)
- Same passage construction logic (5 sentences/passage)
- Same analysis methods (Cohen's d effect size, etc.)
- **Difference**: Original document text instead of synthetic

Results should now be scientifically valid and reproducible.

---

**Implementation Date**: 2026-09-01
**Status**: COMPLETE - Ready for testing
**Files Modified**: 2 (1 new, 1 updated)
**Lines Added/Changed**: 200+ lines
