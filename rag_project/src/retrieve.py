import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic

# Load environment variables (like ANTHROPIC_API_KEY)
load_dotenv()

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Lazy-loaded models to avoid loading on import
_embedding_model = None
_chroma_collection = None
_anthropic_client = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (bge-small-en-v1.5) for retrieval...")
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedding_model

def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(f"Chroma DB directory not found at {CHROMA_DIR}. Run embed.py first.")
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = chroma_client.get_collection(name="cheatsheet_rag")
    return _chroma_collection

def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[WARNING] ANTHROPIC_API_KEY environment variable not set. Claude generation will be unavailable.")
        return None
    # Recreate the client on each call to prevent connection closed/thread reuse issues
    return anthropic.Anthropic(api_key=api_key)

def retrieve_chunks(query, k=4):
    """
    Embeds query and performs similarity search in Chroma DB.
    """
    model = get_embedding_model()
    collection = get_chroma_collection()

    # Embed query
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    # Reformat results
    chunks = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0]*len(docs)
        ids = results["ids"][0]
        
        for i in range(len(docs)):
            chunks.append({
                "chunk_id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": distances[i]
            })
            
    return chunks

def build_grounded_prompt(query, chunks):
    """
    Build the prompt containing retrieved context and the user query.
    """
    context_str = ""
    for idx, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source = meta.get("source_file", "Unknown Source")
        page = meta.get("page", "Unknown Page")
        section = meta.get("section", "General")
        
        context_str += f"--- CONTEXT BLOCK {idx + 1} (Source: {source}, Page: {page}, Section: {section}) ---\n"
        context_str += f"{chunk['text']}\n\n"

    system_prompt = (
        "You are a helpful data science tutor and reference assistant.\n"
        "Your task is to answer the user's question using ONLY the provided context blocks. Do not use outside knowledge.\n"
        "Rules:\n"
        "1. Ground all claims directly in the context.\n"
        "2. If the context does not contain the answer, state clearly: 'I cannot find the answer in the provided cheatsheet context.'\n"
        "3. Always cite which cheat sheet (source file), page, and section the information came from. Use clear markdown citations like [Pandas Cheatsheet, Page 2, groupby].\n"
        "4. Keep your answer clear, concise, and structured."
    )

    user_message = (
        f"Context:\n{context_str}\n"
        f"Question: {query}"
    )

    return system_prompt, user_message

def generate_answer(query, chunks):
    """
    Calls Claude API to answer the query grounded in the retrieved chunks.
    """
    client = get_anthropic_client()
    if not client:
        return "Generation skipped: ANTHROPIC_API_KEY environment variable is not set. Below are the retrieved context blocks:\n\n" + \
               "\n".join([f"- From {c['metadata']['source_file']} (Page {c['metadata']['page']}, Section {c['metadata']['section']}):\n{c['text']}" for c in chunks])

    system_prompt, user_message = build_grounded_prompt(query, chunks)

    print("Calling Claude API...")
    with client:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

    return response.content[0].text

def ask_question(query, k=4):
    """
    End-to-end question answering.
    """
    chunks = retrieve_chunks(query, k)
    if not chunks:
        return {
            "answer": "No relevant context chunks could be retrieved from the vector store.",
            "sources": []
        }
        
    answer = generate_answer(query, chunks)
    
    # Extract unique sources
    sources = []
    seen = set()
    for c in chunks:
        meta = c["metadata"]
        src_key = (meta["source_file"], meta["page"], meta["section"])
        if src_key not in seen:
            seen.add(src_key)
            sources.append({
                "source_file": meta["source_file"],
                "page": meta["page"],
                "section": meta["section"]
            })
            
    return {
        "answer": answer,
        "sources": sources,
        "raw_chunks": chunks
    }

if __name__ == "__main__":
    # Test query
    import sys
    test_query = sys.argv[1] if len(sys.argv) > 1 else "How do I perform a groupby in pandas?"
    print(f"Testing Query: '{test_query}'")
    try:
        res = ask_question(test_query)
        print("\n=== ANSWER ===")
        print(res["answer"])
        print("\n=== SOURCES ===")
        for s in res["sources"]:
            print(f"- {s['source_file']} (Page {s['page']}, Section: {s['section']})")
    except Exception as e:
        print(f"[ERROR] Failed to run retrieval pipeline: {e}")
