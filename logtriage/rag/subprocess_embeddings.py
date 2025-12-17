"""Subprocess-based embedding service for complete memory isolation."""

import logging
import json
import sys
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

def subprocess_embed_text(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                         device: str = "cpu") -> Optional[List[float]]:
    """Generate embedding in a subprocess for complete memory isolation."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        # Generate embedding
        embedding = model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Convert to list for JSON serialization
        if embedding.size > 0:
            return embedding[0].tolist()
        else:
            return None
            
    except Exception as e:
        logger.error(f"Subprocess embedding failed: {e}")
        return None
    finally:
        # Force cleanup in subprocess
        import gc
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

def main():
    """Main function for subprocess execution."""
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Usage: subprocess_embeddings.py <text> <model_name> <device>"}))
        sys.exit(1)
    
    text = sys.argv[1]
    model_name = sys.argv[2]
    device = sys.argv[3]
    
    # Generate embedding
    result = subprocess_embed_text(text, model_name, device)
    
    if result is not None:
        print(json.dumps({"success": True, "embedding": result}))
    else:
        print(json.dumps({"success": False, "error": "Failed to generate embedding"}))

if __name__ == "__main__":
    main()
