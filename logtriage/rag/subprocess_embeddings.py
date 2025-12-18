"""Subprocess-based embedding service for complete memory isolation."""

import logging
import json
import sys
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

def subprocess_embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                          device: str = "cpu", batch_size: int = 4) -> Optional[List[List[float]]]:
    """Generate embeddings in a subprocess for complete memory isolation."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        all_embeddings = []
        
        # Process in batches within subprocess for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Convert to list for JSON serialization
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())
        
        return all_embeddings
        
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
    if len(sys.argv) != 5:
        print(json.dumps({"error": "Usage: subprocess_embeddings.py <texts_json> <model_name> <device> <batch_size>"}))
        sys.exit(1)
    
    texts_json = sys.argv[1]
    model_name = sys.argv[2]
    device = sys.argv[3]
    batch_size = int(sys.argv[4])
    
    try:
        # Parse texts from JSON
        texts = json.loads(texts_json)
        
        if not isinstance(texts, list):
            print(json.dumps({"success": False, "error": "texts must be a list"}))
            sys.exit(1)
        
        # Generate embeddings
        result = subprocess_embed_texts(texts, model_name, device, batch_size)
        
        if result is not None:
            print(json.dumps({"success": True, "embeddings": result}))
        else:
            print(json.dumps({"success": False, "error": "Failed to generate embeddings"}))
            
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main()
