from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticAnalyzer:
    
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Load a pre-trained model for semantic analysis
    
    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy = True
        )
    
    def similarity(self, text1, text2):
        embeddings1 = self.encode([text1, text2])
        
        score = cosine_similarity([embeddings1[0]], [embeddings1[1]])[0][0]
       
        return float(score)         