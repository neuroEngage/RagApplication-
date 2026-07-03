import os
import sys

# Force HuggingFace Hub to run in offline mode (uses local cache, avoids network/httpx issues)
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import chromadb
import torch

# Configure PyTorch CPU threads for faster execution on multi-core systems
torch.set_num_threads(12)

# Reconfigure stdout/stderr to use UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"
MANIFEST_FILE = DATA_DIR / "embedding_manifest.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(file_path):
    """Calculate SHA256 of a file."""
    if not file_path.exists():
        return None
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    # 1. Check chunks file
    if not CHUNKS_FILE.exists():
        print(f"[ERROR] Chunks file not found: {CHUNKS_FILE}. Run chunk.py first.")
        return

    # 2. Load existing embedding manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    # 3. Read all chunks and group by source file
    print("Reading chunks...")
    chunks_by_file = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            source = chunk["source_file"]
            if source not in chunks_by_file:
                chunks_by_file[source] = []
            chunks_by_file[source].append(chunk)

    # 4. Determine which files need embedding
    files_to_embed = []
    new_manifest = {}
    
    import re
    for source, chunks in chunks_by_file.items():
        # Get hash of the extracted text file
        # Derive unique safe_name based on the relative path (source)
        rel_path_str = str(source).replace("\\", "_").replace("/", "_")
        if rel_path_str.lower().endswith(".pdf"):
            rel_path_str = rel_path_str[:-4]
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', rel_path_str)
        extracted_file = DATA_DIR / "extracted" / f"{safe_name}_extracted.txt"
        
        file_hash = get_file_hash(extracted_file)
        if not file_hash:
            print(f"[WARNING] Extracted file not found for {source}, skipping.")
            continue
            
        old_hash = manifest.get(source)
        if old_hash != file_hash:
            files_to_embed.append((source, chunks, file_hash))
            print(f"[CHANGE DETECTED] {source} will be embedded (old hash: {old_hash[:8] if old_hash else None}, new hash: {file_hash[:8]})")
        else:
            # Unchanged
            new_manifest[source] = file_hash
            print(f"[UNCHANGED] {source} (hash: {file_hash[:8]}) - skipping embedding.")

    if not files_to_embed:
        print("\nAll files are up to date. No embedding needed.")
        return

    # 5. Load model local/cache
    print("\nLoading sentence-transformer model (bge-small-en-v1.5)...")
    # This downloads the model to local HuggingFace cache and loads it
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 6. Initialize Chroma DB client
    print("Initializing Chroma DB...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name="cheatsheet_rag",
        metadata={"hnsw:space": "cosine"}  # BGE models work well with cosine similarity
    )

    # 7. Embed and index
    for source, chunks, file_hash in files_to_embed:
        print(f"\nProcessing {source} ({len(chunks)} chunks)...")
        
        # Delete old entries for this file from Chroma
        try:
            collection.delete(where={"source_file": source})
            print(f"  Cleaned existing database entries for {source}")
        except Exception as e:
            print(f"  No existing entries or error cleaning: {e}")

        # Batch embedding
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{
            "source_file": c["source_file"],
            "topic": c["topic"],
            "page": c["page"],
            "section": c["section"]
        } for c in chunks]
        
        print("  Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()
        
        print("  Adding to Chroma...")
        # Chroma supports batch add
        # We can upload in batches of 100 to be safe
        batch_size = 100
        for idx in range(0, len(chunks), batch_size):
            end_idx = min(idx + batch_size, len(chunks))
            collection.add(
                ids=ids[idx:end_idx],
                embeddings=embeddings[idx:end_idx],
                metadatas=metadatas[idx:end_idx],
                documents=texts[idx:end_idx]
            )
            
        new_manifest[source] = file_hash
        
        # Save manifest incrementally to prevent progress loss if interrupted
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(new_manifest, f, indent=4)
            
    print(f"\nEmbedding and indexing complete. Manifest saved to {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
