from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np

class SemanticAnalyzer:
    
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Load a pre-trained model for semantic analysis
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
    
    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy = True
        )
    
    def encode_passages(self, passages, batch_size=32):
        """
        Encode passages with batch processing and statistics tracking.
        
        Args:
            passages (list): List of passage dictionaries with 'text' key
            batch_size (int): Number of passages to encode per batch
            
        Returns:
            dict: {
                'passages': list of passages with 'embedding' added,
                'stats': {
                    'total_passages': int,
                    'embedding_dimension': int,
                    'processing_time_seconds': float,
                    'batch_size': int,
                    'num_batches': int
                }
            }
        """
        start_time = time.time()
        
        # Extract texts for encoding
        texts = [passage['text'] for passage in passages]
        
        # Encode in batches
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeddings = self.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Add embeddings to passages
        passages_with_embeddings = []
        for passage, embedding in zip(passages, embeddings):
            passage_copy = passage.copy()
            passage_copy['embedding'] = embedding
            passages_with_embeddings.append(passage_copy)
        
        processing_time = time.time() - start_time
        
        # Compute statistics
        stats = {
            'total_passages': len(passages),
            'embedding_dimension': self.embedding_dimension,
            'processing_time_seconds': round(processing_time, 4),
            'batch_size': batch_size,
            'num_batches': num_batches,
            'texts_per_second': round(len(texts) / processing_time, 2) if processing_time > 0 else 0
        }
        
        return {
            'passages': passages_with_embeddings,
            'stats': stats
        }
    
    def similarity(self, text1, text2):
        embeddings1 = self.encode([text1, text2])
        
        score = cosine_similarity([embeddings1[0]], [embeddings1[1]])[0][0]
       
        return float(score)         