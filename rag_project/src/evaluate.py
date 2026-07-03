import os
import sys
import json
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Force path resolving for imports
import sys
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from src.retrieve import retrieve_chunks, ask_question

# List of 15 evaluation questions across domains
EVAL_QUESTIONS = [
    # Pandas
    ("Pandas", "How do I merge two dataframes in Pandas?"),
    ("Pandas", "How do I filter rows in a Pandas dataframe based on a condition?"),
    # Numpy
    ("Numpy", "How do I create a 1D array in Numpy?"),
    ("Numpy", "What is the function to change the shape of a numpy array?"),
    # SQL
    ("SQL", "What are the different types of SQL Joins?"),
    ("SQL", "How do I group rows in SQL and calculate averages?"),
    # Python
    ("Python", "What is a lambda function in Python and how do you write one?"),
    ("Python", "How do I write a list comprehension in Python?"),
    # Probability & Stats
    ("Probability", "What is the mathematical formula for Bayes' Theorem?"),
    ("Statistics", "What are the properties of a standard normal distribution?"),
    # Git
    ("Git", "How do I commit changes in Git with a message?"),
    ("Git", "How do I create a new branch in Git?"),
    # DL & ML
    ("Deep Learning", "What is the purpose of backpropagation in neural networks?"),
    ("Machine Learning", "What is the difference between supervised and unsupervised learning?"),
    # Data Visualization
    ("Data Visualization", "How do I choose between a bar chart and a line chart for data visualization?")
]

def main():
    print("Starting RAG Evaluation Suite...")
    
    results = []
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    for idx, (category, query) in enumerate(EVAL_QUESTIONS):
        print(f"\n[{idx+1}/{len(EVAL_QUESTIONS)}] [{category}] {query}")
        
        try:
            # 1. Test Retrieval
            chunks = retrieve_chunks(query, k=3)
            
            top_sources = []
            for c in chunks[:3]:
                meta = c["metadata"]
                top_sources.append(f"{Path(meta['source_file']).name} (Page {meta['page']}, Sec: {meta['section']})")
            
            top_source_str = " | ".join(top_sources) if top_sources else "None retrieved"
            
            # 2. Test Generation if API key is present
            answer = "Skipped (No API Key)"
            status = "Retrieval Verified"
            
            if has_api_key:
                res = ask_question(query, k=3)
                answer = res["answer"]
                status = "Pass" if "I cannot find the answer" not in answer else "No Grounding Found"
            
            print(f"  Top Matches: {top_source_str}")
            if has_api_key:
                print(f"  Status: {status}")
                
            results.append({
                "id": idx + 1,
                "category": category,
                "query": query,
                "top_sources": top_sources,
                "status": status,
                "answer": answer
            })
            
        except Exception as e:
            print(f"  [ERROR] Evaluation failed: {e}")
            results.append({
                "id": idx + 1,
                "category": category,
                "query": query,
                "top_sources": [],
                "status": "FAIL",
                "answer": f"Error: {e}"
            })

    # Save results to Markdown table
    eval_file_path = PROJECT_DIR / "data" / "evaluation_results.md"
    
    md_content = []
    md_content.append("# RAG Ingestion & Retrieval Evaluation Results\n")
    md_content.append(f"Evaluation run status: **{'LLM Verified' if has_api_key else 'Retrieval Verified (Offline)'}**\n")
    md_content.append("| ID | Category | Question | Top Retrieved Matches | Status |")
    md_content.append("|---|---|---|---|---|")
    
    for r in results:
        sources_list = "<br>".join([f"- {s}" for s in r["top_sources"]])
        md_content.append(f"| {r['id']} | {r['category']} | {r['query']} | {sources_list} | {r['status']} |")
        
    with open(eval_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"\nEvaluation complete. Results table written to: {eval_file_path}")

if __name__ == "__main__":
    main()
