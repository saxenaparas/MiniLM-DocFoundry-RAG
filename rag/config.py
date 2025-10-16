# Auto-generated config for Paras_Saxena_Task2
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
RERANK_K = 5
REFUSAL_COSINE_THRESHOLD = 0.3
MODEL_CACHE_DIR = "models"
RANDOM_SEED = 42
VECTOR_STORE = "chroma"  # "chroma" primary; code has graceful fallback
CHROMA_DIR = "data/index/chroma"
SIMILARITY = "cosine"  # "cosine"
