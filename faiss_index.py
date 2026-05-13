import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

FAISS_INDEX_PATH = Path("data") / "faiss_index"


class FAISSIndexManager:
    """Manage FAISS index for semantic search."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.memory_ids = []  # Map from FAISS index position to daily_memory_id
        self.index_file = FAISS_INDEX_PATH / "index.faiss"
        self.ids_file = FAISS_INDEX_PATH / "memory_ids.json"
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing FAISS index or create new one."""
        if not faiss:
            print("FAISS not installed. Semantic search will be disabled.")
            return
        
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        
        if self.index_file.exists() and self.ids_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.ids_file, "r") as f:
                    self.memory_ids = json.load(f)
                print(f"Loaded FAISS index with {len(self.memory_ids)} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Creating new one.")
                self._create_new()
        else:
            self._create_new()
    
    def _create_new(self):
        """Create a new FAISS index."""
        if not faiss:
            return
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.memory_ids = []
    
    def add_embedding(self, memory_id: int, embedding: List[float]):
        """Add an embedding to the index."""
        if not faiss or not self.index:
            return
        
        embedding_array = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_array)
        self.memory_ids.append(memory_id)
        self._save()
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors.
        
        Returns:
            List of (memory_id, distance) tuples
        """
        if not faiss or not self.index or len(self.memory_ids) == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(k, len(self.memory_ids)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.memory_ids):
                results.append((self.memory_ids[idx], float(dist)))
        
        return results
    
    def rebuild_from_db(self, memories: List[dict]):
        """Rebuild index from database memories (with embeddings)."""
        if not faiss:
            return
        
        self.index = None
        self.memory_ids = []
        self._create_new()
        
        for memory in memories:
            if memory.get("embedding"):
                self.add_embedding(memory["id"], memory["embedding"])
        
        print(f"Rebuilt FAISS index with {len(self.memory_ids)} vectors")
    
    def _save(self):
        """Persist index to disk."""
        if not faiss or not self.index:
            return
        
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        
        with open(self.ids_file, "w") as f:
            json.dump(self.memory_ids, f)


# Global instance
_manager = None


def get_faiss_manager():
    """Get or create global FAISS manager."""
    global _manager
    if _manager is None:
        _manager = FAISSIndexManager()
    return _manager