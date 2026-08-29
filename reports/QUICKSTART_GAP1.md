# Gap 1 Implementation — Quick Start Guide

## Running the Experiment

### 1. Install Dependencies
```bash
pip install sentence-transformers spacy scikit-learn numpy pandas
```

### 2. Run the Experiment
```bash
cd /workspaces/legal-judgment-summarizer
PYTHONPATH=/workspaces/legal-judgment-summarizer:$PYTHONPATH python backend/app/experiment_gap1.py
```

### Expected Output
- Comprehensive statistics report
- Files saved to `data/experiments/`:
  - `test_judgment_001_passages.json` (passage metadata without embeddings)
  - `test_judgment_001_stats.json` (processing statistics)

---

## Using the PassageBuilder in Your Code

### Basic Usage: Process a Document

```python
from backend.app.services.passage_builder import PassageBuilder

# Create builder
builder = PassageBuilder(sentences_per_passage=5, document_id="my_judgment")

# Load and process document
with open("path/to/judgment.txt", "r") as f:
    judgment_text = f.read()

result = builder.process_document(judgment_text, batch_size=32)

# Access results
passages = result['passages']  # List of passages with embeddings
stats = result['stats']         # Processing statistics

# Each passage contains:
# - document_id: str
# - passage_id: int
# - sentence_start/end: int (original indices)
# - original_position: {'start_char': int, 'end_char': int}
# - text: str
# - embedding: numpy array (384-dim)

print(f"Processed {stats['total_passages']} passages")
print(f"Time: {stats['processing_time_seconds']} seconds")
print(f"Memory: ~{stats['total_passages'] * 2} KB")
```

### Verify Order Preservation

```python
verification = builder.verify_passage_order(passages)
if verification['valid']:
    print(f"✓ Order valid - {verification['num_passages']} passages")
else:
    print("⚠️ Issues found:")
    for issue in verification['issues']:
        print(f"  - {issue}")
```

### Save Passages

```python
# Save with position metadata (embeddings excluded for file size)
builder.save_passages(passages, "output/passages.json", include_embeddings=False)

# Or save with embeddings (large file)
builder.save_passages(passages, "output/passages_full.json", include_embeddings=True)
```

### Just Build Passages (No Encoding)

```python
# Just create passage structure without embeddings
passages = builder.build_passages(judgment_text, document_id="test")

# Each passage:
# {
#     'document_id': 'test',
#     'passage_id': 0,
#     'sentence_start': 0,
#     'sentence_end': 4,
#     'num_sentences': 5,
#     'text': '...',
#     'original_position': {'start_char': 0, 'end_char': 500}
# }
```

### Encode Existing Passages

```python
# If you already have passage objects with 'text' field
my_passages = [
    {'text': 'First passage...'},
    {'text': 'Second passage...'},
    # ...
]

result = builder.encode_passages(my_passages, batch_size=32)
passages_with_embeddings = result['passages']
stats = result['stats']
```

---

## Using SemanticAnalyzer Directly

```python
from backend.app.services.semantic_analyzer import SemanticAnalyzer

analyzer = SemanticAnalyzer()

# Batch encode multiple passages
passages = [
    {'text': 'The court held that...'},
    {'text': 'The appellant contended...'},
    {'text': 'Unrelated text about weather...'}
]

result = analyzer.encode_passages(passages, batch_size=32)

# Result contains:
# - passages: original passages + 'embedding' field
# - stats: {
#     'total_passages': 3,
#     'embedding_dimension': 384,
#     'processing_time_seconds': 0.1234,
#     'batch_size': 32,
#     'num_batches': 1,
#     'texts_per_second': 24.3
#   }

# Compute similarity
similarity = analyzer.similarity(
    "The court held that Section 45 is unconstitutional",
    "It was submitted that the provision violates Article 21"
)
print(f"Similarity: {similarity:.4f}")  # ~0.54 expected
```

---

## Experiment Results at a Glance

| Metric | Value |
|--------|-------|
| Sample judgment size | 38,989 chars / 6,093 words |
| Sentences | 756 |
| Passages | 152 |
| Sentences per passage | 5 |
| Processing time | 4.19 seconds |
| Throughput | 36.25 passages/sec |
| Embedding dimension | 384 |
| Memory estimate | 0.30 MB |
| **Estimated for full judgment (4,562 sentences)** | **~25 seconds / 1.8 MB** |

---

## Adjusting Parameters

### More/Fewer Sentences per Passage

```python
# Create larger passages (10 sentences each)
builder = PassageBuilder(sentences_per_passage=10)

# Pros: Fewer passages, more context per passage
# Cons: May lose fine-grained information
```

### Batch Size for Encoding

```python
# Smaller batch for memory-constrained environments
result = builder.process_document(text, batch_size=8)

# Larger batch for speed
result = builder.process_document(text, batch_size=64)
```

### Custom Embedding Model

```python
# Currently uses: all-MiniLM-L6-v2 (384-dim)
# To change, modify semantic_analyzer.py line:
# self.model = SentenceTransformer("different-model-name")

# Other options:
# - "sentence-transformers/all-mpnet-base-v2" (768-dim, slower)
# - "sentence-transformers/paraphrase-MiniLM-L6-v2" (384-dim)
# - "sentence-transformers/legal-MiniLM-L6-v2" (hypothetical legal version)
```

---

## Troubleshooting

### Issue: ImportError on spacy
```bash
# Install spacy and the English model
pip install spacy
python -m spacy download en_core_web_sm
```

### Issue: CUDA out of memory
```python
# Reduce batch size
result = builder.process_document(text, batch_size=8)

# Or set device to CPU
from backend.app.services.semantic_analyzer import SemanticAnalyzer
analyzer = SemanticAnalyzer()
analyzer.model.to('cpu')
```

### Issue: Very slow sentence segmentation
- Check spacy installation
- Consider caching sentences if processing multiple documents
- Current implementation: ~750 sentences in 4.19 seconds (including embedding)

---

## Integration with Existing Code

### With the FastAPI Endpoint

The experiment works independently. To integrate with the `/upload` endpoint:

```python
# In backend/app/main.py
from backend.app.services.passage_builder import PassageBuilder

@app.post("/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    # ... existing file upload code ...
    
    # New: Build passages
    builder = PassageBuilder(document_id=file.filename)
    result = builder.process_document(text, batch_size=32)
    
    return {
        "filename": file.filename,
        "passages": len(result['passages']),
        "processing_time": result['stats']['processing_time_seconds'],
        "throughput": result['stats']['texts_per_second'],
        # ... additional fields
    }
```

---

## File Locations

- **Implementation**: 
  - `backend/app/services/passage_builder.py`
  - `backend/app/services/semantic_analyzer.py` (extended)
  
- **Experiment**: 
  - `backend/app/experiment_gap1.py`
  
- **Results**:
  - `data/experiments/test_judgment_001_passages.json`
  - `data/experiments/test_judgment_001_stats.json`
  
- **Documentation**:
  - `GAP1_IMPLEMENTATION_REPORT.md` (comprehensive report)

---

## What's Next?

Gap 1 Stage 2 will implement:
1. **Semantic Retrieval**: Find passages related to a query
2. **Semantic Clustering**: Group related passages automatically
3. **Long-range Relationship Detection**: Link arguments to responses across the document

This foundation (passage representation) makes those stages straightforward to build.

---

## Performance Expectations

- **100 passages**: ~3 seconds
- **500 passages**: ~14 seconds
- **1,000 passages**: ~28 seconds
- **4,562 passages (full judgment)**: ~25-30 seconds (scales linearly)

Memory overhead: ~2KB per passage

---

For detailed information, see `GAP1_IMPLEMENTATION_REPORT.md`
