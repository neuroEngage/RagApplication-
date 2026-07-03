import os
import re
import json
import sys
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"

def detect_sections(text):
    """
    Split page text into chunks based on headings or paragraph groups.
    Try to group text under headings where possible.
    """
    lines = text.split("\n")
    chunks = []
    current_header = "General"
    current_paragraph = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Heuristics for a heading:
        # - Short line (< 50 chars)
        # - Doesn't end with a period, comma, or colon
        # - Is either ALL CAPS or Title Case
        # - Or starts with typical list bullet points like "1.", "•", etc.
        is_heading = (
            len(stripped) < 50 and 
            not stripped.endswith(('.', ',', ';', ':')) and 
            (stripped.isupper() or stripped.istitle() or stripped.startswith(('#', '##', '###', '1.', '2.', '3.')))
        )
        
        if is_heading:
            # Save previous chunk if exists
            if current_paragraph:
                chunks.append({
                    "section": current_header,
                    "text": "\n".join(current_paragraph).strip()
                })
                current_paragraph = []
            current_header = stripped.lstrip('#').strip()
        else:
            current_paragraph.append(line)
            
    if current_paragraph:
        chunks.append({
            "section": current_header,
            "text": "\n".join(current_paragraph).strip()
        })
        
    return chunks

def chunk_file(file_manifest):
    rel_path = file_manifest["file"]
    extracted_rel = file_manifest["output_file"]
    extracted_path = PROJECT_DIR / extracted_rel
    
    if not extracted_path.exists():
        print(f"[ERROR] Extracted text not found: {extracted_path}")
        return []
        
    with open(extracted_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by pages
    page_splits = re.split(r"--- PAGE (\d+) ---\n", content)
    
    # page_splits[0] is empty or text before PAGE 1
    # then page_splits[1] is page number (as string), page_splits[2] is page content, etc.
    chunks = []
    
    path_obj = Path(rel_path)
    if len(path_obj.parts) > 1:
        topic = path_obj.parts[0]
    else:
        topic = "General"
    
    for i in range(1, len(page_splits), 2):
        page_num = int(page_splits[i])
        page_text = page_splits[i+1]
        
        # Detect sections on this page
        sections = detect_sections(page_text)
        
        for idx, sec in enumerate(sections):
            text_content = sec["text"]
            if not text_content or len(text_content) < 20:
                continue
                
            # If chunk is too large, split it further by length (e.g. 600 chars)
            max_chunk_len = 600
            if len(text_content) > max_chunk_len:
                words = text_content.split()
                sub_chunks = []
                temp = []
                temp_len = 0
                for w in words:
                    temp.append(w)
                    temp_len += len(w) + 1
                    if temp_len > max_chunk_len:
                        sub_chunks.append(" ".join(temp))
                        temp = []
                        temp_len = 0
                if temp:
                    sub_chunks.append(" ".join(temp))
            else:
                sub_chunks = [text_content]
                
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunk_id = f"{Path(rel_path).stem}_p{page_num}_s{idx}_{sub_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_file": rel_path,
                    "topic": topic,
                    "page": page_num,
                    "section": sec["section"],
                    "text": sub_text
                })
                
    return chunks

def main():
    manifest_path = DATA_DIR / "extracted_manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}. Run extract.py first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    all_chunks = []
    for item in manifest:
        file_chunks = chunk_file(item)
        all_chunks.extend(file_chunks)
        print(f"Chunked {item['file']}: created {len(file_chunks)} chunks.")
        
    chunks_jsonl_path = DATA_DIR / "chunks.jsonl"
    with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
            
    print(f"\nChunking complete. Total chunks: {len(all_chunks)}")
    print(f"Saved to {chunks_jsonl_path}")

if __name__ == "__main__":
    main()
