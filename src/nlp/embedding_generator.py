"""
Embedding Generator for Crypto Whitepaper Analysis

Generates sentence embeddings using Ollama with nomic-embed-text model.
Designed for efficient batch processing of whitepaper chunks.

Embeddings are used to construct the claims tensor for alignment testing.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

# HTTP requests for Ollama API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Optional: sentence-transformers as fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    text: str
    embedding: np.ndarray
    model: str
    dimension: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class DocumentEmbeddings:
    """All embeddings for a document."""
    project_symbol: str
    project_name: str
    embeddings: List[EmbeddingResult]
    model: str
    dimension: int
    
    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)
    
    @property
    def embedding_matrix(self) -> np.ndarray:
        """Get embeddings as matrix (n_chunks x dimension)."""
        return np.vstack([e.embedding for e in self.embeddings])
    
    @property
    def mean_embedding(self) -> np.ndarray:
        """Get mean embedding across all chunks."""
        return self.embedding_matrix.mean(axis=0)


class EmbeddingGenerator:
    """Generate embeddings using Ollama or sentence-transformers."""
    
    # Ollama embedding models
    OLLAMA_MODELS = {
        "nomic-embed-text": {"dimension": 768, "context": 8192},
        "mxbai-embed-large": {"dimension": 1024, "context": 512},
        "all-minilm": {"dimension": 384, "context": 256},
    }
    
    # Sentence-transformers models (fallback)
    ST_MODELS = {
        "all-MiniLM-L6-v2": {"dimension": 384},
        "all-mpnet-base-v2": {"dimension": 768},
        "paraphrase-multilingual-MiniLM-L12-v2": {"dimension": 384},
    }
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        use_ollama: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Model name (Ollama or sentence-transformers)
            ollama_url: Ollama API URL
            use_ollama: Whether to use Ollama (vs sentence-transformers)
            batch_size: Batch size for embedding generation
        """
        self.model = model
        self.ollama_url = ollama_url
        self.use_ollama = use_ollama
        self.batch_size = batch_size
        
        # Initialize backend
        if use_ollama:
            if not HAS_REQUESTS:
                raise ImportError("requests required for Ollama. Run: pip install requests")
            self._check_ollama_connection()
            self.dimension = self.OLLAMA_MODELS.get(model, {}).get("dimension", 768)
        else:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers required. Run: pip install sentence-transformers")
            self.st_model = SentenceTransformer(model)
            self.dimension = self.ST_MODELS.get(model, {}).get("dimension", 768)
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if self.model not in model_names:
                    print(f"Warning: Model '{self.model}' not found in Ollama.")
                    print(f"Available models: {model_names}")
                    print(f"To install: ollama pull {self.model}")
                    return False
                return True
        except requests.exceptions.ConnectionError:
            print(f"Warning: Cannot connect to Ollama at {self.ollama_url}")
            print("Make sure Ollama is running: ollama serve")
            return False
        except Exception as e:
            print(f"Warning: Ollama check failed: {e}")
            return False
        return False
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            EmbeddingResult with embedding vector
        """
        if self.use_ollama:
            embedding = self._embed_ollama(text)
        else:
            embedding = self._embed_sentence_transformer(text)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimension=len(embedding),
        )
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress indicator
        
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                progress = (i + len(batch)) / total * 100
                print(f"  Embedding batch {i//self.batch_size + 1}: {progress:.1f}% complete")
            
            for text in batch:
                result = self.embed_text(text)
                results.append(result)
            
            # Small delay to avoid overwhelming Ollama
            if self.use_ollama and i + self.batch_size < total:
                time.sleep(0.1)
        
        return results
    
    def embed_document(
        self,
        chunks: List[str],
        project_symbol: str,
        project_name: str,
        show_progress: bool = True
    ) -> DocumentEmbeddings:
        """
        Generate embeddings for all chunks of a document.
        
        Args:
            chunks: List of text chunks
            project_symbol: Crypto ticker
            project_name: Full project name
            show_progress: Show progress indicator
        
        Returns:
            DocumentEmbeddings with all chunk embeddings
        """
        if show_progress:
            print(f"Embedding {len(chunks)} chunks for {project_name}...")
        
        embeddings = self.embed_texts(chunks, show_progress=show_progress)
        
        return DocumentEmbeddings(
            project_symbol=project_symbol,
            project_name=project_name,
            embeddings=embeddings,
            model=self.model,
            dimension=self.dimension,
        )
    
    def _embed_ollama(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["embedding"], dtype=np.float32)
                return embedding
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama API timeout. Try reducing text length.")
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")
    
    def _embed_sentence_transformer(self, text: str) -> np.ndarray:
        """Generate embedding using sentence-transformers."""
        embedding = self.st_model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_similar_chunks(
        self,
        query_embedding: np.ndarray,
        document_embeddings: DocumentEmbeddings,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document to search
            top_k: Number of results to return
        
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        similarities = []
        
        for emb_result in document_embeddings.embeddings:
            sim = self.compute_similarity(query_embedding, emb_result.embedding)
            similarities.append((emb_result.text, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(
        self,
        doc_embeddings: DocumentEmbeddings,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save document embeddings to file.
        
        Args:
            doc_embeddings: DocumentEmbeddings to save
            output_path: Output file path (.npz format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy compressed array
        np.savez_compressed(
            output_path,
            embeddings=doc_embeddings.embedding_matrix,
            texts=np.array([e.text for e in doc_embeddings.embeddings], dtype=object),
            metadata=np.array({
                "project_symbol": doc_embeddings.project_symbol,
                "project_name": doc_embeddings.project_name,
                "model": doc_embeddings.model,
                "dimension": doc_embeddings.dimension,
                "num_embeddings": doc_embeddings.num_embeddings,
            })
        )
    
    def load_embeddings(
        self,
        input_path: Union[str, Path]
    ) -> DocumentEmbeddings:
        """
        Load document embeddings from file.
        
        Args:
            input_path: Input file path (.npz format)
        
        Returns:
            DocumentEmbeddings object
        """
        data = np.load(input_path, allow_pickle=True)
        
        embeddings = data["embeddings"]
        texts = data["texts"]
        metadata = data["metadata"].item()
        
        emb_results = [
            EmbeddingResult(
                text=str(text),
                embedding=embeddings[i],
                model=metadata["model"],
                dimension=metadata["dimension"],
            )
            for i, text in enumerate(texts)
        ]
        
        return DocumentEmbeddings(
            project_symbol=metadata["project_symbol"],
            project_name=metadata["project_name"],
            embeddings=emb_results,
            model=metadata["model"],
            dimension=metadata["dimension"],
        )


def create_mock_embeddings(
    texts: List[str],
    dimension: int = 768
) -> List[np.ndarray]:
    """
    Create mock embeddings for testing without Ollama.
    
    Uses random vectors with text-based seeds for reproducibility.
    """
    embeddings = []
    for text in texts:
        # Use hash of text as seed for reproducibility
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        embedding = np.random.randn(dimension).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    return embeddings


if __name__ == "__main__":
    print("=== Embedding Generator Test ===\n")
    
    # Test with mock embeddings (no Ollama required)
    print("Testing mock embeddings...")
    
    sample_texts = [
        "Bitcoin is a peer-to-peer electronic cash system.",
        "Ethereum enables smart contracts and decentralized applications.",
        "Solana provides high throughput for decentralized applications.",
        "Chainlink provides decentralized oracle services.",
        "Monero focuses on privacy and anonymity.",
    ]
    
    mock_embs = create_mock_embeddings(sample_texts)
    
    print(f"Generated {len(mock_embs)} embeddings")
    print(f"Embedding dimension: {mock_embs[0].shape[0]}")
    
    # Test similarity
    print("\n=== Similarity Test ===")
    for i, text in enumerate(sample_texts):
        for j, text2 in enumerate(sample_texts):
            if i < j:
                sim = np.dot(mock_embs[i], mock_embs[j])
                print(f"  {sample_texts[i][:30]}... vs {sample_texts[j][:30]}...: {sim:.3f}")
    
    # Test with real Ollama if available
    print("\n=== Ollama Connection Test ===")
    try:
        generator = EmbeddingGenerator(model="nomic-embed-text", use_ollama=True)
        print("✓ Ollama connection successful")
        
        # Generate single embedding
        result = generator.embed_text(sample_texts[0])
        print(f"✓ Generated embedding: {result.dimension} dimensions")
        
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        print("  Using mock embeddings for development")
