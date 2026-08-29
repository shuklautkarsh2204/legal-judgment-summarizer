"""
Passage Builder for Long-Document Understanding (Gap 1)

Builds structured passage representations from long legal judgments while preserving:
- Original document position
- Sentence boundaries
- Document structure for relationship tracing
"""

from pathlib import Path
from backend.app.services.preprocessing import split_into_sentences
from backend.app.services.semantic_analyzer import SemanticAnalyzer
import json


class PassageBuilder:
    """
    Builds manageable passages from long documents while preserving position information.
    """
    
    def __init__(self, sentences_per_passage=5, document_id=None):
        """
        Initialize PassageBuilder.
        
        Args:
            sentences_per_passage (int): Number of sentences to group into one passage
            document_id (str): Optional document identifier
        """
        self.sentences_per_passage = sentences_per_passage
        self.document_id = document_id
        self.semantic_analyzer = SemanticAnalyzer()
    
    def load_document(self, document_path):
        """
        Load text from a document file.
        
        Args:
            document_path (Path or str): Path to the document text file
            
        Returns:
            str: Document text
        """
        document_path = Path(document_path)
        with open(document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _get_sentence_spans(self, text, sentences):
        """
        Compute the exact character spans of each sentence in document order.

        This avoids naive text.find()/rfind() calls, which fail when the same sentence
        text appears multiple times in the document. We move forward through the
        original text in order, using the previous sentence end as the search start.
        """
        spans = []
        search_start = 0

        for sentence in sentences:
            sentence_text = sentence.strip()
            if not sentence_text:
                continue

            start_char = text.find(sentence_text, search_start)
            if start_char == -1:
                # Conservative fallback: search the rest of the document from the
                # earliest remaining location if the sentence was not found after the
                # previous offset. This preserves order while keeping the mapping
                # consistent with the text as it appears in the document.
                raise ValueError(
                    f"Sentence text not found after character position "
                    f"{search_start}: {sentence_text[:80]!r}"
                )

            end_char = start_char + len(sentence_text)
            spans.append({
                'text': sentence_text,
                'start_char': start_char,
                'end_char': end_char
            })
            search_start = end_char

        return spans

    def build_passages(self, text, document_id=None):
        """
        Build structured passages from document text.
        
        Process:
        1. Split text into sentences
        2. Group sentences into passages
        3. Preserve original position information
        4. Create structured passage representation
        
        Args:
            text (str): Full document text
            document_id (str): Optional document ID (overrides constructor default)
            
        Returns:
            list: List of passage dictionaries with structure:
                {
                    'document_id': str,
                    'passage_id': int,
                    'sentence_start': int,  # Original sentence index
                    'sentence_end': int,    # Original sentence index (inclusive)
                    'num_sentences': int,
                    'text': str,
                    'original_position': {'start_char': int, 'end_char': int}
                }
        """
        doc_id = document_id or self.document_id or "unknown"
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        if not sentences:
            return []

        sentence_spans = self._get_sentence_spans(text, sentences)
        passages = []
        passage_id = 0
        
        # Group sentences into passages
        for i in range(0, len(sentence_spans), self.sentences_per_passage):
            sentence_batch = sentence_spans[i:i + self.sentences_per_passage]
            sentence_start_idx = i
            sentence_end_idx = i + len(sentence_batch) - 1
            
            # Join sentences to form passage text while preserving document order.
            passage_text = " ".join(span['text'] for span in sentence_batch)

            first_span = sentence_batch[0]
            last_span = sentence_batch[-1]
            passage = {
                'document_id': doc_id,
                'passage_id': passage_id,
                'sentence_start': sentence_start_idx,
                'sentence_end': sentence_end_idx,
                'num_sentences': len(sentence_batch),
                'text': passage_text,
                'original_position': {
                    'start_char': first_span['start_char'],
                    'end_char': last_span['end_char']
                }
            }
            
            passages.append(passage)
            passage_id += 1
        
        return passages
    
    def encode_passages(self, passages, batch_size=32):
        """
        Encode passages using semantic analyzer.
        
        Args:
            passages (list): List of passage dictionaries
            batch_size (int): Batch size for encoding
            
        Returns:
            dict: {
                'passages': passages with embeddings,
                'stats': encoding statistics
            }
        """
        return self.semantic_analyzer.encode_passages(passages, batch_size=batch_size)
    
    def process_document(self, text, document_id=None, batch_size=32):
        """
        End-to-end processing: build passages and encode them.
        
        Args:
            text (str): Document text
            document_id (str): Optional document ID
            batch_size (int): Batch size for encoding
            
        Returns:
            dict: {
                'passages': passages with embeddings,
                'stats': combined statistics,
                'document_stats': document-level statistics
            }
        """
        # Build passages
        passages = self.build_passages(text, document_id=document_id)
        
        # Document-level statistics
        doc_stats = {
            'total_passages': len(passages),
            'sentences_per_passage': self.sentences_per_passage,
            'total_sentences': sum(p['num_sentences'] for p in passages),
            'total_characters': len(text),
            'total_words': len(text.split()),
            'document_id': document_id or self.document_id or "unknown"
        }
        
        # Encode passages
        encoding_result = self.encode_passages(passages, batch_size=batch_size)
        
        # Combine statistics
        stats = {
            **encoding_result['stats'],
            **doc_stats
        }
        
        return {
            'passages': encoding_result['passages'],
            'stats': stats
        }
    
    def save_passages(self, passages, output_path, include_embeddings=False):
        """
        Save passages to a JSON file.
        
        Args:
            passages (list): Passages to save
            output_path (Path or str): Output file path
            include_embeddings (bool): Whether to include embeddings in saved file
                                      (warning: makes file very large)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare passages for JSON serialization
        passages_to_save = []
        for passage in passages:
            p = passage.copy()
            if not include_embeddings and 'embedding' in p:
                # Store embedding shape info instead of full embedding
                p['embedding_shape'] = list(p['embedding'].shape)
                del p['embedding']
            elif 'embedding' in p:
                # Convert numpy array to list for JSON serialization
                p['embedding'] = p['embedding'].tolist()
            passages_to_save.append(p)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(passages_to_save, f, indent=2, ensure_ascii=False)
    
    def verify_passage_order(self, passages):
        """
        Verify that passages maintain document order.
        
        Args:
            passages (list): List of passages
            
        Returns:
            dict: Verification results
        """
        if not passages:
            return {'valid': True, 'num_passages': 0}
        
        issues = []
        
        # Check passage_id continuity
        for i, passage in enumerate(passages):
            if passage['passage_id'] != i:
                issues.append(f"Passage ID mismatch at index {i}: expected {i}, got {passage['passage_id']}")
            
            # Check sentence indices are increasing
            if i > 0:
                prev_passage = passages[i - 1]
                if passage['sentence_start'] <= prev_passage['sentence_end']:
                    issues.append(f"Sentence order break between passage {i-1} and {i}")
        
        return {
            'valid': len(issues) == 0,
            'num_passages': len(passages),
            'issues': issues
        }
    
    def test_repeated_sentences_have_correct_positions():
    text = (
        "The appellant filed an appeal. "
        "The Court considered the matter. "
        "The appellant filed an appeal."
    )

    builder = PassageBuilder(
        sentences_per_passage=1,
        document_id="repeated_test"
    )

    passages = builder.build_passages(text)

    assert len(passages) == 3

    assert passages[0]["original_position"]["start_char"] == 0

    assert (
        passages[2]["original_position"]["start_char"]
        > passages[0]["original_position"]["start_char"]
    )    
